/* get ground truth tensorflow cpu wrapper
 * By Zhaoyu SU, Email: zsuad@connect.ust.hk
 * All Rights Reserved. Sep, 2019.
 */
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("GetBboxOp")
    .Input("roi_attrs: float32")
    .Input("gt_bbox: float32")
    .Input("input_num_list: int32")
    .Output("bbox: float32")
    .Output("bbox_conf: int32")
    .Output("bbox_cls: int32")
    .Attr("expand_ratio: float")
    .Attr("cls_num: int")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle roi_base_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &roi_base_coors_shape));

        int cls_num;
        TF_RETURN_IF_ERROR(c->GetAttr("cls_num", &cls_num));

        DimensionHandle npoint = c->Dim(roi_base_coors_shape, 0);
        ShapeHandle bbox_shape = c->MakeShape({npoint, 7});
        ShapeHandle bbox_conf_shape = c->MakeShape({npoint});
        ShapeHandle bbox_cls_shape = c->MakeShape({npoint, cls_num});
        c->set_output(0, bbox_shape);
        c->set_output(1, bbox_conf_shape);
        c->set_output(2, bbox_cls_shape);

        return Status::OK();

    }); // InferenceContext

void get_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr, float expand_ratio, int cls_num,
                           const float* roi_attrs,
                           const float* gt_bbox,
                           const int* input_num_list,
                           int* input_accu_list,
                           float* bbox,
                           int* bbox_conf,
                           int* bbox_cls);

class GetBboxOp: public OpKernel {
public:
    explicit GetBboxOp(OpKernelConstruction* context): OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("cls_num", &cls_num));
        OP_REQUIRES_OK(context, context->GetAttr("expand_ratio", &expand_ratio));
    }
    void Compute(OpKernelContext* context) override {

        const Tensor& roi_attrs = context->input(0);
        auto roi_attrs_ptr = roi_attrs.template flat<float>().data();
        OP_REQUIRES(context, roi_attrs.dims() == 2 && roi_attrs.dim_size(1) == 7,
            errors::InvalidArgument("Expect input roi_attrs has shape [npoints, 7]"));

        const Tensor& gt_bbox = context->input(1);
        auto gt_bbox_ptr = gt_bbox.template flat<float>().data();
        OP_REQUIRES(context, gt_bbox.dim_size(2) == 8,
                    errors::InvalidArgument("Attribute of bbox has to be 9: [w, l, h, x, y, z, r, cls]."));

        const Tensor& input_num_list = context->input(2);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims() == 1,
                    errors::InvalidArgument("FPS Op expects input in shape: [batch_size]."));

        int batch_size = input_num_list.dim_size(0);
        int npoint = roi_attrs.dim_size(0);
        int bbox_attr = gt_bbox.dim_size(2);
        int nbbox = gt_bbox.dim_size(1);

        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemset(input_accu_list_ptr, 0, batch_size * sizeof(int));

        Tensor* bbox = nullptr;
        auto bbox_shape = TensorShape({npoint, 7});
        OP_REQUIRES_OK(context, context->allocate_output(0, bbox_shape, &bbox));
        float* bbox_ptr = bbox->template flat<float>().data();
        cudaMemset(bbox_ptr, 0.f, npoint * 7 * sizeof(float));

        Tensor* bbox_conf = nullptr;
        auto bbox_conf_shape = TensorShape({npoint});
        OP_REQUIRES_OK(context, context->allocate_output(1, bbox_conf_shape, &bbox_conf));
        int* bbox_conf_ptr = bbox_conf->template flat<int>().data();
        cudaMemset(bbox_conf_ptr, 0, npoint * sizeof(int));

        Tensor* bbox_cls = nullptr;
        auto bbox_cls_shape = TensorShape({npoint, cls_num});
        OP_REQUIRES_OK(context, context->allocate_output(2, bbox_cls_shape, &bbox_cls));
        int* bbox_cls_ptr = bbox_cls->template flat<int>().data();
        cudaMemset(bbox_cls_ptr, 0, npoint * cls_num * sizeof(int));

        get_bbox_gpu_launcher(batch_size, npoint, nbbox, bbox_attr, expand_ratio, cls_num,
                                     roi_attrs_ptr,
                                     gt_bbox_ptr,
                                     input_num_list_ptr,
                                     input_accu_list_ptr,
                                     bbox_ptr,
                                     bbox_conf_ptr,
                                     bbox_cls_ptr);
    }
private:
    int cls_num;
    float expand_ratio;
};
REGISTER_KERNEL_BUILDER(Name("GetBboxOp").Device(DEVICE_GPU), GetBboxOp);