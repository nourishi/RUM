import logging

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.rum_model import (
    MotionRelocalizer,
    UncertaintyMemoryScheduler,
    VisualPrototypeMemory,
)
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # language-free uncertainty/prototype/motion adapters
        use_language_free_vos: bool = False,
        lfm_num_part_prototypes: int = 4,
        lfm_prototype_momentum: float = 0.95,
        lfm_write_uncertainty_thresh: float = 0.35,
        lfm_relocalize_uncertainty_thresh: float = 0.65,
        lfm_motion_prior_strength: float = 3.0,
        lfm_motion_prior_sigma: float = 0.12,
        lfm_motion_max_delta: float = 0.2,
        lfm_uncertainty_ema: float = 0.5,
        lfm_topk_reid_candidates: int = 3,
        lfm_relocalize_consecutive_frames: int = 3,
        lfm_relocalize_cooldown: int = 3,
        lfm_relocalize_center_consistency_thresh: float = 0.2,
        lfm_control_temperature: float = 0.7,
        lfm_short_memory_momentum: float = 0.6,
        lfm_long_memory_momentum: float = 0.98,
        lfm_motion_min_region_scale: float = 0.06,
        lfm_motion_max_region_scale: float = 0.35,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        # Part 5: language-free tracking adapters (uncertainty/prototype/motion)
        self.use_language_free_vos = use_language_free_vos
        self.lfm_uncertainty_ema = lfm_uncertainty_ema
        self.lfm_topk_reid_candidates = lfm_topk_reid_candidates
        self.lfm_num_part_prototypes = lfm_num_part_prototypes
        self.lfm_relocalize_consecutive_frames = max(
            int(lfm_relocalize_consecutive_frames), 1
        )
        self.lfm_relocalize_cooldown = max(int(lfm_relocalize_cooldown), 0)
        self.lfm_relocalize_center_consistency_thresh = max(
            float(lfm_relocalize_center_consistency_thresh), 0.0
        )
        self.usm_scheduler = None
        self.visual_prototype_memory = None
        self.motion_relocalizer = None
        if self.use_language_free_vos:
            self.usm_scheduler = UncertaintyMemoryScheduler(
                hidden_dim=self.hidden_dim,
                write_threshold=lfm_write_uncertainty_thresh,
                relocalize_threshold=lfm_relocalize_uncertainty_thresh,
                control_temperature=lfm_control_temperature,
            )
            self.visual_prototype_memory = VisualPrototypeMemory(
                hidden_dim=self.hidden_dim,
                num_parts=lfm_num_part_prototypes,
                momentum=lfm_prototype_momentum,
                short_momentum=lfm_short_memory_momentum,
                long_momentum=lfm_long_memory_momentum,
            )
            self.motion_relocalizer = MotionRelocalizer(
                hidden_dim=self.hidden_dim,
                prior_strength=lfm_motion_prior_strength,
                prior_sigma=lfm_motion_prior_sigma,
                max_delta=lfm_motion_max_delta,
                min_region_scale=lfm_motion_min_region_scale,
                max_region_scale=lfm_motion_max_region_scale,
            )

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                if self.use_language_free_vos:
                    lfm_state = output_dict.get("_lfm_state", None)
                    if lfm_state is not None and lfm_state["proto_valid"].any():
                        proto_ptr = self.visual_prototype_memory.get_memory_pointer(
                            lfm_state
                        )
                        if proto_ptr is not None:
                            pos_and_ptrs.append(
                                (0, proto_ptr.to(device, non_blocking=True))
                            )
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _init_lfm_state(self, output_dict, batch_size, device):
        if not self.use_language_free_vos:
            return None

        state = output_dict.get("_lfm_state", None)
        reset_state = (
            state is None
            or state["global_proto"].size(0) != batch_size
            or state["global_proto"].device != device
        )
        if reset_state:
            default_region_scale = (
                self.motion_relocalizer.min_region_scale
                if self.motion_relocalizer is not None
                else 0.1
            )
            state = {
                "global_proto": torch.zeros(batch_size, self.hidden_dim, device=device),
                "part_proto": torch.zeros(
                    batch_size,
                    self.lfm_num_part_prototypes,
                    self.hidden_dim,
                    device=device,
                ),
                "proto_valid": torch.zeros(
                    batch_size, 1, device=device, dtype=torch.bool
                ),
                "short_global_proto": torch.zeros(
                    batch_size, self.hidden_dim, device=device
                ),
                "short_part_proto": torch.zeros(
                    batch_size,
                    self.lfm_num_part_prototypes,
                    self.hidden_dim,
                    device=device,
                ),
                "short_valid": torch.zeros(
                    batch_size, 1, device=device, dtype=torch.bool
                ),
                "long_global_proto": torch.zeros(
                    batch_size, self.hidden_dim, device=device
                ),
                "long_part_proto": torch.zeros(
                    batch_size,
                    self.lfm_num_part_prototypes,
                    self.hidden_dim,
                    device=device,
                ),
                "long_valid": torch.zeros(
                    batch_size, 1, device=device, dtype=torch.bool
                ),
                "last_center": torch.zeros(batch_size, 2, device=device),
                "prev_center": torch.zeros(batch_size, 2, device=device),
                "last_size": torch.full(
                    (batch_size, 2), default_region_scale, device=device
                ),
                "prev_size": torch.full(
                    (batch_size, 2), default_region_scale, device=device
                ),
                "center_valid": torch.zeros(
                    batch_size, 1, device=device, dtype=torch.bool
                ),
                "ema_uncertainty": torch.zeros(batch_size, 1, device=device),
                "high_uncertainty_streak": torch.zeros(
                    batch_size, 1, device=device, dtype=torch.long
                ),
                "relocalize_cooldown": torch.zeros(
                    batch_size, 1, device=device, dtype=torch.long
                ),
                "write_hits": torch.zeros(1, device=device),
                "freeze_hits": torch.zeros(1, device=device),
                "relocalize_hits": torch.zeros(1, device=device),
                "relocalize_total": torch.zeros(1, device=device),
                "consistency_reject_hits": torch.zeros(1, device=device),
                "consistency_reject_total": torch.zeros(1, device=device),
                "uncertainty_sum": torch.zeros(1, device=device),
                "uncertainty_sq_sum": torch.zeros(1, device=device),
                "uncertainty_count": torch.zeros(1, device=device),
                "uncertainty_min": torch.ones(1, device=device),
                "uncertainty_max": torch.zeros(1, device=device),
            }
            output_dict["_lfm_state"] = state
        else:
            for k, v in state.items():
                state[k] = v.to(device, non_blocking=True)
        return state

    def _collect_reid_candidate_centers(self, output_dict, obj_ptr):
        if (
            not self.use_language_free_vos
            or self.motion_relocalizer is None
            or self.lfm_topk_reid_candidates <= 0
        ):
            return None, None, None

        candidate_ptrs = []
        candidate_centers = []
        candidate_sizes = []
        frame_outputs = list(output_dict["cond_frame_outputs"].values()) + list(
            output_dict["non_cond_frame_outputs"].values()
        )
        for out in frame_outputs:
            if "obj_ptr" not in out or "pred_masks" not in out:
                continue
            if out["obj_ptr"] is None or out["pred_masks"] is None:
                continue
            ptr = out["obj_ptr"].to(obj_ptr.device, non_blocking=True).detach()
            center, size = self.motion_relocalizer.compute_mask_region(
                out["pred_masks"].to(obj_ptr.device, non_blocking=True).detach()
            )
            candidate_ptrs.append(ptr)
            candidate_centers.append(center)
            candidate_sizes.append(size)

        if len(candidate_ptrs) == 0:
            return None, None, None

        candidate_ptrs = torch.stack(candidate_ptrs, dim=1)
        candidate_centers = torch.stack(candidate_centers, dim=1)
        candidate_sizes = torch.stack(candidate_sizes, dim=1)
        cur_ptr = F.normalize(obj_ptr.float(), dim=-1).unsqueeze(1)
        ptr_scores = torch.sum(
            cur_ptr * F.normalize(candidate_ptrs.float(), dim=-1), dim=-1
        )
        topk = min(self.lfm_topk_reid_candidates, ptr_scores.size(1))
        top_scores, top_indices = torch.topk(ptr_scores, k=topk, dim=1)
        top_centers = candidate_centers.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        top_sizes = candidate_sizes.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        return top_centers, top_sizes, top_scores

    def _update_relocalize_gate(
        self,
        state,
        base_relocalize_mask,
        emergency_relocalize_mask=None,
    ):
        cooldown_active = state["relocalize_cooldown"] > 0
        streak = torch.where(
            base_relocalize_mask & (~cooldown_active),
            state["high_uncertainty_streak"] + 1,
            torch.zeros_like(state["high_uncertainty_streak"]),
        )
        state["high_uncertainty_streak"] = streak
        gated_mask = streak >= self.lfm_relocalize_consecutive_frames
        if emergency_relocalize_mask is not None:
            gated_mask = gated_mask | (emergency_relocalize_mask & (~cooldown_active))
        return gated_mask

    def _apply_language_free_modules(
        self,
        frame_idx,
        is_init_cond_frame,
        output_dict,
        pix_feat,
        ious,
        low_res_masks,
        high_res_masks,
        obj_ptr,
        object_score_logits,
    ):
        if not self.use_language_free_vos:
            return {
                "low_res_masks": low_res_masks,
                "high_res_masks": high_res_masks,
                "obj_ptr": obj_ptr,
                "object_score_logits": object_score_logits,
                "high_res_masks_for_mem_enc": high_res_masks,
                "uncertainty": None,
                "proto_similarity": None,
                "write_prob": None,
                "freeze_prob": None,
                "relocalize_prob": None,
                "write_mask": None,
                "relocalize_mask": None,
            }

        state = self._init_lfm_state(output_dict, obj_ptr.size(0), obj_ptr.device)
        cur_global_proto, cur_part_proto = self.visual_prototype_memory.extract(
            obj_ptr=obj_ptr,
            pix_feat=pix_feat,
            pred_mask_logits=low_res_masks,
        )
        proto_similarity = self.visual_prototype_memory.similarity(
            state, obj_ptr, cur_part_proto
        )

        if self.pred_obj_scores:
            obj_conf = object_score_logits.sigmoid().clamp(0.0, 1.0)
        else:
            obj_conf = torch.sigmoid(low_res_masks.flatten(1).mean(dim=1, keepdim=True))
        iou_conf = ious.float()
        if iou_conf.dim() == 1:
            iou_conf = iou_conf.unsqueeze(1)
        elif iou_conf.dim() > 2:
            iou_conf = iou_conf.view(iou_conf.size(0), -1)
        iou_conf = iou_conf.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)

        uncertainty, write_prob, freeze_prob, relocalize_prob = self.usm_scheduler(
            obj_conf=obj_conf,
            iou_conf=iou_conf,
            proto_conf=proto_similarity,
        )

        if self.lfm_uncertainty_ema > 0:
            uncertainty = (
                self.lfm_uncertainty_ema * state["ema_uncertainty"]
                + (1.0 - self.lfm_uncertainty_ema) * uncertainty
            )
        state["ema_uncertainty"] = uncertainty.detach()

        write_threshold = self.usm_scheduler.write_threshold
        relocalize_threshold = self.usm_scheduler.relocalize_threshold
        midpoint_threshold = 0.5 * (write_threshold + relocalize_threshold)
        emergency_threshold = min(relocalize_threshold + 0.10, 0.90)

        # Use uncertainty as the primary hard control signal.
        # Soft action probabilities only modulate strength, so relocalization
        # is not starved when write_prob dominates early in training.
        base_relocalize_mask = (uncertainty >= relocalize_threshold) | (
            (relocalize_prob >= 0.20) & (uncertainty >= midpoint_threshold)
        )
        emergency_relocalize_mask = uncertainty >= emergency_threshold
        write_mask = (uncertainty <= write_threshold) & (~base_relocalize_mask)
        freeze_mask = (~write_mask) & (~base_relocalize_mask)
        relocalize_mask = self._update_relocalize_gate(
            state,
            base_relocalize_mask,
            emergency_relocalize_mask=emergency_relocalize_mask,
        )

        obj_ptr = self.visual_prototype_memory.refine_obj_ptr(obj_ptr, state, uncertainty)

        target_center = None
        target_size = None
        target_valid = torch.zeros_like(relocalize_mask)
        relocalize_weight = torch.where(
            base_relocalize_mask,
            torch.clamp(0.40 + 0.60 * torch.maximum(relocalize_prob, uncertainty), max=1.0),
            relocalize_prob.clone(),
        )
        motion_center = None
        motion_size = None
        motion_valid = state["center_valid"]
        motion_conf = torch.zeros_like(uncertainty)
        if motion_valid.any():
            motion_center, motion_size = self.motion_relocalizer.predict_region(
                state["last_center"],
                state["prev_center"],
                state["last_size"],
                state["prev_size"],
            )
            center_delta = torch.norm(
                state["last_center"] - state["prev_center"], dim=-1, keepdim=True
            )
            size_delta = torch.norm(
                state["last_size"] - state["prev_size"], dim=-1, keepdim=True
            )
            center_ratio = (
                center_delta / max(self.motion_relocalizer.max_delta, 1e-6)
            ).clamp(0.0, 1.0)
            size_ratio = (
                size_delta / max(self.motion_relocalizer.max_region_scale, 1e-6)
            ).clamp(0.0, 1.0)
            stability = (1.0 - 0.5 * center_ratio - 0.5 * size_ratio).clamp(0.0, 1.0)
            motion_conf = motion_valid.float() * (0.1 + 0.9 * stability)

        reid_center = None
        reid_size = None
        reid_valid = torch.zeros_like(relocalize_mask)
        reid_conf = torch.zeros_like(uncertainty)
        consistency_reject_mask = torch.zeros_like(relocalize_mask)
        consistency_score = torch.ones_like(uncertainty)
        candidate_centers, candidate_sizes, candidate_scores = self._collect_reid_candidate_centers(
            output_dict, obj_ptr
        )
        if candidate_centers is not None:
            candidate_weight_logits = torch.softmax(candidate_scores, dim=1)
            candidate_weights = candidate_weight_logits.unsqueeze(-1)
            reid_center = torch.sum(candidate_centers * candidate_weights, dim=1)
            reid_size = torch.sum(candidate_sizes * candidate_weights, dim=1)
            normalized_scores = (0.5 * (candidate_scores + 1.0)).clamp(0.0, 1.0)
            reid_conf = torch.sum(
                candidate_weight_logits * normalized_scores, dim=1, keepdim=True
            )
            reid_valid = reid_conf > 0.0

        if motion_center is not None and reid_center is not None:
            both_valid = motion_valid & reid_valid
            center_distance = torch.norm(
                motion_center - reid_center, dim=-1, keepdim=True
            )
            consistency_score = torch.exp(
                -center_distance
                / max(self.lfm_relocalize_center_consistency_thresh, 1e-6)
            ).clamp(0.0, 1.0)
            consistency_reject_mask = both_valid & (consistency_score < 0.10)
            conf_sum = (motion_conf + reid_conf).clamp_min(1e-6)
            fused_center = (
                motion_conf * motion_center + reid_conf * reid_center
            ) / conf_sum
            fused_size = (motion_conf * motion_size + reid_conf * reid_size) / conf_sum
            prefer_motion = motion_conf >= reid_conf
            fallback_center = torch.where(
                prefer_motion.expand(-1, 2), motion_center, reid_center
            )
            fallback_size = torch.where(
                prefer_motion.expand(-1, 2), motion_size, reid_size
            )
            blended_center = consistency_score.expand(-1, 2) * fused_center + (
                1.0 - consistency_score
            ).expand(-1, 2) * fallback_center
            blended_size = consistency_score.expand(-1, 2) * fused_size + (
                1.0 - consistency_score
            ).expand(-1, 2) * fallback_size

            target_center = torch.zeros_like(motion_center)
            target_size = torch.zeros_like(motion_size)
            target_center = torch.where(
                both_valid.expand(-1, 2), blended_center, target_center
            )
            target_size = torch.where(
                both_valid.expand(-1, 2), blended_size, target_size
            )
            target_valid = target_valid | both_valid
            soft_consistency = torch.where(
                consistency_reject_mask,
                torch.full_like(consistency_score, 0.25),
                0.50 + 0.50 * consistency_score,
            )
            relocalize_weight = torch.where(
                both_valid,
                relocalize_weight * soft_consistency,
                relocalize_weight,
            )

            motion_only = motion_valid & (~reid_valid)
            target_center = torch.where(
                motion_only.expand(-1, 2), motion_center, target_center
            )
            target_size = torch.where(
                motion_only.expand(-1, 2), motion_size, target_size
            )
            target_valid = target_valid | motion_only

            reid_only = reid_valid & (~motion_valid)
            target_center = torch.where(
                reid_only.expand(-1, 2), reid_center, target_center
            )
            target_size = torch.where(
                reid_only.expand(-1, 2), reid_size, target_size
            )
            target_valid = target_valid | reid_only
        elif motion_center is not None:
            target_center = motion_center
            target_size = motion_size
            target_valid = motion_valid
        elif reid_center is not None:
            target_center = reid_center
            target_size = reid_size
            target_valid = reid_valid

        relocalize_mask = relocalize_mask & target_valid
        if relocalize_mask.any() and target_center is not None and target_size is not None:
            low_res_relocalized = self.motion_relocalizer.apply_motion_prior(
                low_res_masks,
                target_center,
                target_size,
                uncertainty,
                relocalize_weight=relocalize_weight,
            )
            high_res_relocalized = self.motion_relocalizer.apply_motion_prior(
                high_res_masks,
                target_center,
                target_size,
                uncertainty,
                relocalize_weight=relocalize_weight,
            )
            relocalize_hw = relocalize_mask.unsqueeze(-1).unsqueeze(-1)
            low_res_masks = torch.where(relocalize_hw, low_res_relocalized, low_res_masks)
            high_res_masks = torch.where(
                relocalize_hw, high_res_relocalized, high_res_masks
            )
            state["high_uncertainty_streak"] = torch.where(
                relocalize_mask,
                torch.zeros_like(state["high_uncertainty_streak"]),
                state["high_uncertainty_streak"],
            )

        high_res_masks_for_mem_enc = high_res_masks
        if not is_init_cond_frame:
            base_write_weight = torch.clamp(0.35 + 0.65 * write_prob, min=0.35, max=1.0)
            relocalize_write_weight = torch.clamp(
                0.55 + 0.45 * torch.maximum(relocalize_prob, uncertainty),
                min=0.55,
                max=1.0,
            )
            effective_write_weight = torch.where(
                relocalize_mask,
                relocalize_write_weight,
                base_write_weight,
            )
            effective_write_weight = torch.where(
                freeze_mask,
                torch.clamp(effective_write_weight, min=0.45),
                effective_write_weight,
            )
            high_res_masks_for_mem_enc = (
                effective_write_weight.unsqueeze(-1).unsqueeze(-1) * high_res_masks
            )
        else:
            effective_write_weight = torch.ones_like(write_prob)

        state["relocalize_total"] += relocalize_mask.new_tensor(
            [relocalize_mask.numel()], dtype=torch.float32
        )
        state["write_hits"] += effective_write_weight.sum().detach()
        state["freeze_hits"] += freeze_mask.float().sum().detach()
        state["relocalize_hits"] += relocalize_mask.float().sum().detach()
        state["consistency_reject_hits"] += (
            consistency_reject_mask.float().sum().detach()
        )
        state["consistency_reject_total"] += consistency_reject_mask.new_tensor(
            [consistency_reject_mask.numel()], dtype=torch.float32
        )
        state["uncertainty_sum"] += uncertainty.sum().detach()
        state["uncertainty_sq_sum"] += (uncertainty * uncertainty).sum().detach()
        state["uncertainty_count"] += uncertainty.new_tensor(
            [uncertainty.numel()], dtype=torch.float32
        )
        state["uncertainty_min"] = torch.minimum(
            state["uncertainty_min"], uncertainty.min().detach().view(1)
        )
        state["uncertainty_max"] = torch.maximum(
            state["uncertainty_max"], uncertainty.max().detach().view(1)
        )
        cooldown = torch.clamp(state["relocalize_cooldown"] - 1, min=0)
        if self.lfm_relocalize_cooldown > 0:
            cooldown = torch.where(
                relocalize_mask,
                torch.full_like(cooldown, self.lfm_relocalize_cooldown),
                cooldown,
            )
        state["relocalize_cooldown"] = cooldown

        if not self.training:
            total = state["relocalize_total"].clamp_min(1.0)
            write_rate = state["write_hits"] / total
            freeze_rate = state["freeze_hits"] / total
            relocalize_rate = state["relocalize_hits"] / total
            consistency_reject_rate = (
                state["consistency_reject_hits"]
                / state["consistency_reject_total"].clamp_min(1.0)
            )
            uncertainty_mean = state["uncertainty_sum"] / state["uncertainty_count"].clamp_min(
                1.0
            )
            uncertainty_var = (
                state["uncertainty_sq_sum"]
                / state["uncertainty_count"].clamp_min(1.0)
                - uncertainty_mean * uncertainty_mean
            ).clamp_min(0.0)
            logging.info(
                "LFM stats frame=%s write=%.4f freeze=%.4f relocalize=%.4f consistency_reject=%.4f uncertainty(mean=%.4f,std=%.4f,min=%.4f,max=%.4f)",
                frame_idx,
                write_rate.item(),
                freeze_rate.item(),
                relocalize_rate.item(),
                consistency_reject_rate.item(),
                uncertainty_mean.item(),
                uncertainty_var.sqrt().item(),
                state["uncertainty_min"].item(),
                state["uncertainty_max"].item(),
            )

        update_weight = effective_write_weight if not is_init_cond_frame else torch.ones_like(write_prob)
        self.visual_prototype_memory.update(
            state,
            cur_global_proto.detach(),
            cur_part_proto.detach(),
            update_weight,
            uncertainty.detach(),
        )

        current_center, current_size = self.motion_relocalizer.compute_mask_region(
            high_res_masks.detach()
        )
        center_valid = state["center_valid"].expand(-1, 2)
        state["prev_center"] = torch.where(
            center_valid, state["last_center"], current_center
        )
        state["last_center"] = current_center
        state["prev_size"] = torch.where(center_valid, state["last_size"], current_size)
        state["last_size"] = current_size
        state["center_valid"] = torch.ones_like(state["center_valid"], dtype=torch.bool)

        return {
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "high_res_masks_for_mem_enc": high_res_masks_for_mem_enc,
            "uncertainty": uncertainty,
            "proto_similarity": proto_similarity,
            "write_prob": write_prob,
            "freeze_prob": freeze_prob,
            "relocalize_prob": relocalize_prob,
            "write_mask": write_mask,
            "relocalize_mask": relocalize_mask,
        }

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
        high_res_masks_for_mem_enc=None,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            if high_res_masks_for_mem_enc is None:
                high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        lfm_outputs = self._apply_language_free_modules(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            output_dict=output_dict,
            pix_feat=pix_feat,
            ious=ious,
            low_res_masks=low_res_masks,
            high_res_masks=high_res_masks,
            obj_ptr=obj_ptr,
            object_score_logits=object_score_logits,
        )
        low_res_masks = lfm_outputs["low_res_masks"]
        high_res_masks = lfm_outputs["high_res_masks"]
        obj_ptr = lfm_outputs["obj_ptr"]
        object_score_logits = lfm_outputs["object_score_logits"]

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if self.use_language_free_vos:
            current_out["lfm_uncertainty"] = lfm_outputs["uncertainty"]
            current_out["lfm_proto_similarity"] = lfm_outputs["proto_similarity"]
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
            high_res_masks_for_mem_enc=lfm_outputs["high_res_masks_for_mem_enc"],
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

