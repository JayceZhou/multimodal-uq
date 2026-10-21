import copy
from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.target_uq import centerhead_stage3_proposals, transfusion_stage3_proposals


from .base import Base3DFusionModel
from .stage3 import Stage3ReliabilityConflict

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        auxiliary_object_head: bool = False,
        freeze_base: bool = False,
        stage3: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])
        if auxiliary_object_head:
            self.heads["aux_object"] = copy.deepcopy(self.heads["object"])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
        if auxiliary_object_head:
            self.loss_scale["aux_object"] = 1.0

        stage3_module = copy.deepcopy(stage3) if stage3 is not None else None
        self.freeze_stage3_base = stage3_module is not None and stage3_module.pop("freeze_base", True)
        self.stage3_fuser_trainable = stage3_module is not None and stage3_module.pop("fuser_trainable", False)
        if stage3_module is not None:
            rng_state = torch.random.get_rng_state()
            self.stage3 = Stage3ReliabilityConflict(**stage3_module)
            torch.random.set_rng_state(rng_state)
        else:
            self.stage3 = None
        self.stage3_last = None

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']


        self.init_weights()
        self.freeze_base = freeze_base
        if freeze_base:
            self._freeze_base()
        elif self.freeze_stage3_base:
            self._freeze_stage3_base()

    def _freeze_base(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("heads.aux_object."))

    def _freeze_stage3_base(self) -> None:
        for name, parameter in self.named_parameters():
            trainable = name.startswith("stage3.")
            if self.stage3_fuser_trainable and name.startswith("fuser."):
                trainable = True
            parameter.requires_grad_(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.encoders.eval()
            if self.fuser is not None:
                self.fuser.eval()
            self.decoder.eval()
            for name, head in self.heads.items():
                if name != "aux_object":
                    head.eval()
        elif self.freeze_stage3_base:
            self.encoders.eval()
            if self.fuser is not None and not self.stage3_fuser_trainable:
                self.fuser.eval()
            self.decoder.eval()
            self.heads.eval()
            self.stage3.train(mode)
        return self

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        auxiliary_losses = {}
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        stage3_maps = None
        if self.stage3 is not None:
            features, stage3_maps = self.stage3(features, metas)
            self.stage3_last = stage3_maps

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if self.freeze_base and type != "aux_object":
                    continue
                if type in ("object", "aux_object"):
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.stage3 is not None:
                stage3_losses = self.stage3.loss(stage3_maps, gt_bboxes_3d, metas)
                self.stage3_object_stats = stage3_losses.pop("object_stats")
                outputs.update(stage3_losses)
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "aux_object":
                    continue
                if type == "object":
                    pred_dict = head(x, metas)
                    stage3_target_uq = None
                    if kwargs.get("return_stage3_target_uq", False):
                        if head.__class__.__name__ == "CenterHead":
                            stage3_target_uq = centerhead_stage3_proposals(
                                pred_dict,
                                head,
                                num_classes=sum(head.num_classes),
                                num_proposals=kwargs.get("stage3_num_proposals", 200),
                            )
                        else:
                            stage3_target_uq = transfusion_stage3_proposals(
                                pred_dict,
                                head.bbox_coder,
                                num_classes=head.num_classes,
                                num_proposals=head.num_proposals,
                                nms_type=head.test_cfg.get("nms_type"),
                            )
                    with_query_provenance = bool(kwargs.get("return_query_provenance", False))
                    with_postprocess_trace = bool(kwargs.get("return_postprocess_trace", False))
                    bboxes = head.get_bboxes(pred_dict, metas, return_query_provenance=with_query_provenance, return_postprocess_trace=with_postprocess_trace)
                    for k, item in enumerate(bboxes):
                        cursor = 3
                        if with_query_provenance:
                            boxes, scores, labels, query_indices = item[:4]
                            cursor = 4
                        else:
                            boxes, scores, labels = item[:3]
                        postprocess_trace = item[cursor] if with_postprocess_trace else None
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                        if with_query_provenance:
                            outputs[k]["query_provenance"] = query_indices.detach().cpu()
                        if with_postprocess_trace:
                            outputs[k]["postprocess_trace"] = {
                                key: None if value is None else value.detach().cpu()
                                for key, value in postprocess_trace.items()
                            }
                        if stage3_target_uq is not None:
                            outputs[k]["stage3_target_uq"] = {
                                "sample_token": metas[k].get("token"),
                                "branch": kwargs.get("stage3_branch", "fusion"),
                                "class_probability_source": stage3_target_uq["class_probability_source"],
                                "class_logits": stage3_target_uq["class_logits"][k].detach().cpu(),
                                "features": stage3_target_uq["features"][k].detach().cpu(),
                                "proposal_ids": stage3_target_uq["proposal_ids"][k].detach().cpu(),
                                "query_indices": stage3_target_uq["query_indices"][k].detach().cpu(),
                                "decoder_query_ids": stage3_target_uq["decoder_query_ids"][k].detach().cpu(),
                                "query_labels": stage3_target_uq["query_labels"][k].detach().cpu(),
                                "post_nms_ids": stage3_target_uq["post_nms_ids"][k].detach().cpu(),
                                "post_score_for_mapping": stage3_target_uq["post_score_for_mapping"][k].detach().cpu(),
                                "decoded_boxes": stage3_target_uq["decoded_boxes"][k].detach().cpu(),
                                "means": {
                                    name: value[k].detach().cpu()
                                    for name, value in stage3_target_uq["means"].items()
                                },
                                "kept_post_nms_mask": stage3_target_uq["kept_post_nms_mask"][k].detach().cpu(),
                            }
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
