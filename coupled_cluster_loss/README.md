 message LayerParameter {
    optional string name = 1; // the layer name
    optional string type = 2; // the layer type
 @@ -396,6 +396,7 @@ message LayerParameter {
    optional SetLossParameter set_loss_param = 142;
    optional ROIPoolingParameter roi_pooling_param = 143;
    optional SmoothL1LossParameter smooth_l1_loss_param = 144;
 +  optional CoupledClusterLossParameter coupled_cluster_loss_param = 145;
  }
 + 
 +message CoupledClusterLossParameter {
 +  optional float margin = 1 [default = 1];
 +  optional int32 group_size = 2 [default = 3];
 +  optional float scale = 3 [default = 1];
 +  optional bool log_flag = 4 [default = false];
 +  // optional int32 pos_num = 3 [default = 1];
 +  // optional int32 neg_num = 4 [default = 1];
 +}
