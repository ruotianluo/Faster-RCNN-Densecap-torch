require 'nn'
require 'densecap.modules.BoxRegressionCriterion'

local crit, parent = torch.class('nn.BoxesRegressionCriterion', 'nn.Criterion')

--------------------------------------------------------------------------------
--[[
A criterion for bounding box regression losses.

For bounding box regression, we always predict transforms on top of anchor boxes.
Instead of directly penalizing the difference between the ground-truth box and
predicted boxes, penalize the difference between the transforms and the optimal
transforms that would have converted the anchor boxes into the ground-truth boxes.

This criterion accepts as input the anchor boxes, transforms, and target boxes;
on the forward pass it uses the anchors and target boxes to compute target tranforms,
and returns the loss between the input transforms and computed target transforms.

On the backward pass we compute the gradient of this loss with respect to both the
input transforms and the input anchor boxes.

Inputs:
- input: A list of:
  - anchor_boxes: Tensor of shape (B, 4) giving anchor box coords as (xc, yc, w, h)
  - transforms: Tensor of shape (B, 4 * num_classes) giving transforms as (tx, ty, tw, th)
  - gt_labels: Tensor of shape (B, 1) giving the label of the bbox
- target_boxes: Tensor of shape (B, 4) giving target boxes as (xc, yc, w, h)
--]]

function crit:__init(w)
  parent.__init(self)
  self.box_reg = nn.BoxRegressionCriterion(w)
end


function crit:clearState()
  self.gradInput[1]:set()
  self.gradInput[2]:set()
  self.transform:set()
end


function crit:updateOutput(input, target_boxes)
  local anchor_boxes, transforms, gt_labels = unpack(input)
  assert(transforms:dim() == 3, 'The bbox transforms should be 3-dim matrix')
  local transforms_view = transforms:view(transforms:size(1), -1)
  local inds = gt_labels.new(gt_labels:size(1), 4)
  for i = 1, 4 do
    inds[{{},{i}}]:copy(torch.mul(gt_labels-1, 4)+i)
  end
  self.transform = transforms_view:gather(2, inds)
  self.output = self.box_reg:forward({anchor_boxes, self.transform}, target_boxes)
  return self.output
end


function crit:updateGradInput(input, target_boxes)
  local anchor_boxes, transforms, gt_labels = unpack(input)
  assert(transforms:dim() == 3, 'The bbox transforms should be 3-dim matrix')
  local inds = gt_labels.new(gt_labels:size(1), 4)
  for i = 1, 4 do
    inds[{{},{i}}]:copy(torch.mul(gt_labels-1, 4)+i)
  end
  local grad_transforms = transforms.new(#transforms):zero()
  local grad_transforms_view = grad_transforms:view(transforms:size(1), -1)
  self.gradInput = self.box_reg:backward({anchor_boxes, self.transform}, target_boxes)
  grad_transforms_view:scatter(2, inds, self.gradInput[2])
  self.gradInput[2] = grad_transforms
  self.gradInput[3] = gt_labels.new()
  return self.gradInput
end
