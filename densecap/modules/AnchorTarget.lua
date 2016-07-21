require 'densecap.modules.BoxIoU'
local box_utils = require 'densecap.box_utils'
local utils = require 'densecap.utils'

local layer, parent = torch.class('nn.AnchorTarget', 'nn.Module')


function layer:__init(options)
  parent.__init(self)
  self.low_thresh = utils.getopt(options, 'low_thresh', 0.3)
  self.high_thresh = utils.getopt(options, 'high_thresh', 0.7)
  self.batch_size = utils.getopt(options, 'batch_size', 256)
  self.fg_fraction = utils.getopt(options, 'fg_fraction', 0.25)
  
  self.output = {{torch.Tensor()}, {torch.Tensor()}, {torch.Tensor()}}
  self.gradInput = {torch.Tensor()}

  self.num_pos, self.num_neg = nil, nil
  self.pos_input_idx = nil
  self.pos_target_idx = nil
  self.neg_input_idx = nil

  self.x_min, self.x_max = nil, nil
  self.y_min, self.y_max = nil, nil

  self.box_iou = nn.BoxIoU()
end

function layer:setBounds(bounds)
  self.x_min = utils.getopt(bounds, 'x_min', nil)
  self.x_max = utils.getopt(bounds, 'x_max', nil)
  self.y_min = utils.getopt(bounds, 'y_min', nil)
  self.y_max = utils.getopt(bounds, 'y_max', nil)
end

local function unpack_dims(input_anchors, target_boxes)
  local N, B1 = input_anchors:size(1), input_anchors:size(2)
  local B2 = target_boxes:size(2)
  
  assert(input_anchors:size(3) == 4 and target_boxes:size(3) == 4)
  assert(target_boxes:size(1) == N)
  
  return N, B1, B2
end


--[[
  Input:
  
  List of two lists. The first list contains data about the input boxes,
  and the second list contains data about the target boxes.

  The first element of the first list is input_anchors, a Tensor of shape (N, B1, 4)
  giving coordinates of the input boxes in (xc, yc, w, h) format.

  All other elements of the first list are tensors of shape (N, B1, Di) parallel to
  input_anchors; Di can be different for each element.

  The first element of the second list is target_boxes, a Tensor of shape (N, B2, 4)
  giving coordinates of the target boxes in (xc, yc, w, h) format.

  All other elements of the second list are tensors of shape (N, B2, Dj) parallel
  to target_boxes; Dj can be different for each Tensor.

  
  Returns a list of three lists:

  The first list contains data about positive input boxes. The first element is of
  shape (P, 4) and contains coordinates of positive boxes; the other elements
  correspond to the additional input data about the input boxes; in particular the
  ith element has shape (P, Di).

  In which the second element is empty.

  The second list contains data about target boxes corresponding to positive
  input boxes. The first element is of shape (P, 4) and contains coordinates of
  target boxes corresponding to sampled positive input boxes; the other elements
  correspond to the additional input data about the target boxes; in particular the
  jth element has shape (P, Dj).

  The third list contains data about negative input boxes. The first element is of
  shape (M, 4) and contains coordinates of negative input boxes; the other elements
  correspond to the additional input data about the input boxes; in particular the
  ith element has shape (M, Di).

  In which the second element is empty.
--]]
function layer:updateOutput(input)
  -- Unpack the input
  local input_data = input[1]
  local target_data = input[2]
  local input_anchors = input_data[2]
  local target_boxes = target_data[1]
  local N, B1, B2 = unpack_dims(input_anchors, target_boxes)
  assert(N == 1, 'Only minibatches of 1 are supported')


  local inbounds_mask = torch.ByteTensor(N, B1):fill(1) -- N x B1

  -- Maybe find the input boxes that fall outside the boundaries
  -- and exclude them from the pos and neg masks
  if self.x_min and self.y_min and self.x_max and self.y_max then
    -- Convert from (xc, yc, w, h) to (x1, y1, x2, y2) format to make
    -- it easier to find boxes that are out of bounds
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(input_anchors)
    local x_min_mask = torch.lt(boxes_x1y1x2y2:select(3, 1), self.x_min):byte()
    local y_min_mask = torch.lt(boxes_x1y1x2y2:select(3, 2), self.y_min):byte()
    local x_max_mask = torch.gt(boxes_x1y1x2y2:select(3, 3), self.x_max):byte()
    local y_max_mask = torch.gt(boxes_x1y1x2y2:select(3, 4), self.y_max):byte()
    inbounds_mask[x_min_mask] = 0
    inbounds_mask[y_min_mask] = 0
    inbounds_mask[x_max_mask] = 0
    inbounds_mask[y_max_mask] = 0
  end

  print(inbounds_mask:sum() .. ' in ' .. B1)
  local inbounds_idx = inbounds_mask:view(-1):nonzero():view(-1) -- The index of in-bound boxes
  -- input_anchors changes to (N, num_inbounds, 4)
  input_anchors = input_anchors:index(2, inbounds_idx)
  local num_inbounds = inbounds_idx:size(1)

  -- #2: Choose the pos and neg

  local ious = self.box_iou:forward({input_anchors, target_boxes}) -- N x num_inbounds x B2
  local input_max_iou, input_idx = ious:max(3)   -- N x num_inbounds x 1
  input_max_iou = input_max_iou:view(N, num_inbounds)
  input_idx = input_idx:view(N, num_inbounds)
  local _, target_idx = ious:max(2) -- N x 1 x B2
  target_idx = target_idx:view(N, B2)

  -- Pick positive and negative boxes based on IoU thresholds
  self.pos_mask = torch.gt(input_max_iou, self.high_thresh) -- N x num_inbounds
  self.neg_mask = torch.lt(input_max_iou, self.low_thresh)  -- N x num_inbounds

  -- Count as positive each input box that has maximal IoU with each target box,
  -- even if it is outside the bounds or does not meet the thresholds.
  -- This is important since things will crash if we don't have at least one
  -- positive box.
  self.pos_mask:scatter(2, target_idx, 1)
  self.neg_mask:scatter(2, target_idx, 0)

  
  self.pos_mask = self.pos_mask:view(num_inbounds):byte()
  self.neg_mask = self.neg_mask:view(num_inbounds):byte()

  if self.neg_mask:sum() == 0 then
    -- There were no negatives; this can happen if all input boxes are either:
    -- (1) An input box with maximal IoU with a target box
    -- (2) Out of bounds, therefore clipped
    -- (3) max IoU to all target boxes is in the range [low_thresh, high_thresh]
    -- This should be a pretty rare case, but we still need to handle it.
    -- Ideally this should do something like sort the non-positive in-bounds boxes
    -- by their max IoU to target boxes and set the negative set to be those with
    -- minimal IoU to target boxes; however this is complicated so instead we'll
    -- just sample from non-positive boxes to get negatives.
    -- We'll also log this event in the __GLOBAL_STATS__ table; if this happens
    -- regularly then we should handle it more cleverly.

    self.neg_mask:mul(self.pos_mask, -1):add(1) -- set neg_mask to inverse of pos_mask
    local k = 'AnchorTarget no negatives'
    local old_val = utils.__GLOBAL_STATS__[k] or 0
    utils.__GLOBAL_STATS__[k] = old_val + 1
  end

  local pos_mask_nonzero = self.pos_mask:nonzero():view(-1)
  local neg_mask_nonzero = self.neg_mask:nonzero():view(-1)

  local total_pos = pos_mask_nonzero:size(1)
  local total_neg = neg_mask_nonzero:size(1)

  self.num_pos = math.min(math.floor(self.batch_size * self.fg_fraction), total_pos)
  self.num_neg = self.batch_size - self.num_pos

  -- We always sample positives without replacemet
  local pos_p = torch.ones(total_pos)
  local pos_sample_idx = torch.multinomial(pos_p, self.num_pos, false)

  -- We sample negatives with replacement if there are not enough negatives
  -- to fill out the minibatch
  local neg_p = torch.ones(total_neg)
  local neg_replace = (total_neg < self.num_neg)
  if neg_replace then
    local k = 'Anchor Target negative with replacement'
    local old_val = utils.__GLOBAL_STATS__[k] or 0
    utils.__GLOBAL_STATS__[k] = old_val + 1
  end
  local neg_sample_idx = torch.multinomial(neg_p, self.num_neg, neg_replace)
  
  if self.debug_pos_sample_idx then
    pos_sample_idx = self.debug_pos_sample_idx
  end
  if self.debug_neg_sample_idx then
    neg_sample_idx = self.debug_neg_sample_idx
  end

  self.pos_input_idx = pos_mask_nonzero:index(1, pos_sample_idx)
  self.pos_target_idx = input_idx:index(2, self.pos_input_idx):view(self.num_pos)
  self.neg_input_idx = neg_mask_nonzero:index(1, neg_sample_idx)

  -- #4: To transform the index back to index before out-of-bound elimination and
  -- nms.
  self.pos_input_idx = inbounds_idx:index(1, self.pos_input_idx)
  self.neg_input_idx = inbounds_idx:index(1, self.neg_input_idx)

  for i = 1, #input_data do
    -- The output tensors for additional data will be lazily instantiated
    -- on the first forward pass, which is probably after the module has been
    -- cast to the right datatype, so we make sure the new Tensors have the
    -- same type as the corresponding elements of the input.
    local dtype = input_data[i]:type()
    if #self.output[1] < i then
      table.insert(self.output[1], torch.Tensor():type(dtype))
    end
    if #self.output[3] < i then
      table.insert(self.output[3], torch.Tensor():type(dtype))
    end
    local D = input_data[i]:size(3)
    self.output[1][i]:resize(self.num_pos, D)
    self.output[3][i]:resize(self.num_neg, D)
  end
  for i = 1, #target_data do
    local dtype = target_data[i]:type()
    if #self.output[2] < i then
      table.insert(self.output[2], torch.Tensor():type(dtype))
    end
    local D = target_data[i]:size(3)
    self.output[2][i]:resize(self.num_pos, D)
  end

  -- Now use the indicies to actually copy data from inputs to outputs
  for _, i in pairs({2, 3, 4}) do
    self.output[1][i]:index(input_data[i], 2, self.pos_input_idx)
    self.output[3][i]:index(input_data[i], 2, self.neg_input_idx)
    -- The call to index adds an extra dimension at the beginning for batch
    -- size, but since its a singleton we just squeeze it out
    local D = input_data[i]:size(3)
    self.output[1][i] = self.output[1][i]:view(self.num_pos, D)
    self.output[3][i] = self.output[3][i]:view(self.num_neg, D)
  end
  for i = 1, 1 do
    self.output[2][i]:index(target_data[i], 2, self.pos_target_idx)
    local D = target_data[i]:size(3)
    self.output[2][i] = self.output[2][i]:view(self.num_pos, D)
  end

  return self.output
end


--[[

Arguments:
  - input: Same as last call to updateOutput.
  - gradOutput: A list of two elements, giving the gradients of output[1] and output[3].

Returns:
A single list, giving the gradients of the loss with respect to the input data
(the first argument to updateOutput)
--]]
function layer:updateGradInput(input, gradOutput)
  -- Unpack the input and gradOutput
  local input_data = input[1]
  local target_data = input[2]
  local input_anchors = input_data[2]
  local target_boxes = target_data[1]
  local N = input_anchors:size(1)
  local B1, B2 = input_anchors:size(2), target_boxes:size(2)
  assert(N == 1, 'Only minibatches of 1 are supported')
  
  -- Resize the gradInput. It should be the same size as input_data.
  -- As in the forward pass, we need to worry about data types since
  -- Tensors are lazily instantiated here.
  for i = 1, #input_data do
    local dtype = input_data[i]:type()
    if #self.gradInput < i then
      table.insert(self.gradInput, torch.Tensor():type(dtype))
    end
    self.gradInput[i]:resizeAs(input_data[i])
  end

  -- Copy the gradients from gradOutput to self.input
  -- This assumes that there is no overlap between the positive and negative samples
  -- coming out of the BoxSampler, which should always be true.
  for i = 1, #input_data do
    self.gradInput[i]:zero()
    
    if gradOutput[1][i] then
      local v1 = gradOutput[1][i]:view(1, self.num_pos, -1)
      self.gradInput[i]:indexCopy(2, self.pos_input_idx, v1)
    end

    if gradOutput[2][i] then
      local v2 = gradOutput[2][i]:view(1, self.num_neg, -1)
      self.gradInput[i]:indexCopy(2, self.neg_input_idx, v2)
    end
    
  end

  return self.gradInput
end

function layer:clearState()
  self.output[1] = {}
  self.output[2] = {}
  self.output[3] = {}
  self.gradInput = {}

  self.num_pos, self.num_neg = nil, nil
  self.pos_input_idx = nil
  self.pos_target_idx = nil
  self.neg_input_idx = nil

  self.box_iou:clearState()
end
