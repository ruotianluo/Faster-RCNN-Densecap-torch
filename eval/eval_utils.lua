
local cjson = require 'cjson'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'

local eval_utils = {}


--[[
Evaluate a DenseCapModel on a split of data from a DataLoader.

Input: An object with the following keys:
- model: A DenseCapModel object to evaluate; required.
- loader: A DataLoader object; required.
- split: Either 'val' or 'test'; default is 'val'
- max_images: Integer giving the number of images to use, or -1 to use the
  entire split. Default is -1.
- id: ID for cross-validation; default is ''.
- dtype: torch datatype to which data should be cast before passing to the
  model. Default is 'torch.FloatTensor'.
--]]
function eval_utils.eval_split(kwargs)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_images = utils.getopt(kwargs, 'max_images', -1)
  local id = utils.getopt(kwargs, 'id', '')
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  local split_to_int = {val=1, test=2}
  split = split_to_int[split]
  print('using split ', split)
  
  model:evaluate()
  loader:resetIterator(split)
  local evaluator = {}
  for cls = 1, model.opt.num_classes do -- 1 is not background, is the output of RPN
    evaluator[cls] = DenseCaptioningEvaluator{id=id}
  end

  local counter = 0
  local all_losses = {}
  while true do
    counter = counter + 1
    
    -- Grab a batch of data and convert it to the right dtype
    local data = {}
    local loader_kwargs = {split=split, iterate=true}
    local img, gt_boxes, gt_labels, info, _ = loader:getBatch(loader_kwargs)
    local data = {
      image = img:type(dtype),
      gt_boxes = gt_boxes:type(dtype),
      gt_labels = gt_labels:type(dtype),
    }
    info = info[1] -- Since we are only using a single image

    -- Call forward_backward to compute losses
    model.timing = false
    model.dump_vars = false
    model.cnn_backward = false
    local losses = model:forward_backward(data)
    table.insert(all_losses, losses)

    -- Call forward_test to make predictions, and pass them to evaluator
    local boxes, scores = model:forward_test(data.image)
    -- 1 is the RPN output
    evaluator[1]:addResult(scores[1], boxes[1], 
          gt_boxes[1], 'RPN')
    for cls = 2, model.opt.num_classes do
      local sel_inds = torch.range(1,gt_labels[1]:size(1))[gt_labels[1]:eq(cls)]:long()
      local cls_gt_boxes
      if sel_inds:numel() ~= 0 then
        cls_gt_boxes = gt_boxes[1]:index(1, sel_inds)
      end
      evaluator[cls]:addResult(scores[cls], boxes[cls], -- table index start from 1
          cls_gt_boxes, model.opt.idx_to_cls[cls])
    end
    
    -- Print a message to the console
    local msg = 'Processed image %s (%d / %d) of split %d, detected %d regions'
    local num_images = info.split_bounds[2]
    if max_images > 0 then num_images = math.min(num_images, max_images) end
    local num_boxes = 0; for i = 1, #boxes do num_boxes = num_boxes + boxes[i]:size(1) end
    print(string.format(msg, info.filename, counter, num_images, split, num_boxes))

    -- Break out if we have processed enough images
    if max_images > 0 and counter >= max_images then break end
    if info.split_bounds[1] == info.split_bounds[2] then break end
  end

  local loss_results = utils.dict_average(all_losses)
  print('Loss stats:')
  print(loss_results)
  print('Average loss: ', loss_results.total_loss)
  
  local ap_results = {}
  for cls = 1, model.opt.num_classes do
    ap_results[cls] = evaluator[cls]:evaluate()
  end
  ap_results.rpn_ap = ap_results[1]['ov0.5'] -- RPN ap
  ap_results.map = {}
  for cls = 2, model.opt.num_classes do
    table.insert(ap_results.map, ap_results[cls]['ov0.5'])
  end
  ap_results.map = utils.average_values(ap_results.map)

  print(string.format('rpn AP: %f', 100 * ap_results.rpn_ap))
  print(string.format('mAP: %f', 100 * ap_results.map))
  
  local out = {
    loss_results=loss_results,
    ap_results=ap_results,
  }
  return out
end

function eval_utils.score_labels(records)
  -- serialize records to json file
  local blob = {}
  blob.scores = {}
  for k,v in pairs(records) do
    local c = v.candidate
    local r = v.references[1]
    if c == r then
      table.insert(blob.scores, 1)
    else
      table.insert(blob.scores, 0)
    end
  end
  return blob
end


local function pluck_boxes(ix, boxes, text)
  -- ix is a list (length N) of LongTensors giving indices to boxes/text. Use them to do merge
  -- this is done because multiple ground truth annotations can be on top of each other, and
  -- we want to instead group many overlapping boxes into one, with multiple caption references.
  -- return boxes Nx4, and text[] of length N

  local N = #ix
  local new_boxes = torch.zeros(N, 4)
  local new_text = {}

  for i=1,N do
    
    local ixi = ix[i]
    local n = ixi:nElement()
    local bsub = boxes:index(1, ixi)
    local newbox = torch.mean(bsub, 1)
    new_boxes[i] = newbox

    local texts = {}
    if text then
      for j=1,n do
        table.insert(texts, text[ixi[j]])
      end
    end
    table.insert(new_text, texts)
  end

  return new_boxes, new_text
end


local DenseCaptioningEvaluator = torch.class('DenseCaptioningEvaluator')
function DenseCaptioningEvaluator:__init(opt)
  self.all_scores = {}
  self.records = {}
  self.n = 1
  self.npos = 0
  self.id = utils.getopt(opt, 'id', '')
end

-- boxes is (B x 4) are xcycwh, scores are (B, ), target_boxes are (M x 4) also as xcycwh.
-- these can be both on CPU or on GPU (they will be shipped to CPU if not already so)
-- predict_text is length B list of strings, target_text is length M list of strings.
function DenseCaptioningEvaluator:addResult(scores, boxes, target_boxes, class)
  assert(scores:size(1) == boxes:size(1))
  assert(boxes:nDimension() == 2)

  -- convert both boxes to x1y1x2y2 coordinate systems
  boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
  if target_boxes == nil then
    target_boxes = boxes.new()
  else
    target_boxes = box_utils.xcycwh_to_x1y1x2y2(target_boxes)
  end

  -- make sure we're on CPU
  boxes = boxes:float()
  scores = scores:double() -- grab the positives class (1)
  target_boxes = target_boxes:float()

  -- merge ground truth boxes that overlap by >= 0.7
  --local mergeix = box_utils.merge_boxes(target_boxes, 0.7) -- merge groups of boxes together
  --local merged_boxes, merged_text = pluck_boxes(mergeix, target_boxes, target_text)
  --merged_boxes = target_boxes
  --merged_text = {}
  --for k,v in pairs(target_text) do table.insert(merged_text, {v}) end

  -- 1. Sort detections by decreasing confidence
  local Y,IX = torch.sort(scores,1,true) -- true makes order descending
  
  local nd = scores:size(1) -- number of detections
  local nt = 0
  if target_boxes:numel() ~= 0 then
    nt = target_boxes:size(1) -- number of gt boxes
  end
  local used = torch.zeros(nt)
  for d=1,nd do -- for each detection in descending order of confidence
    local ii = IX[d]
    local bb = boxes[ii]
    
    -- assign the box to its best match in true boxes
    local ovmax = 0
    local jmax = -1
    for j=1,nt do
      local bbgt = target_boxes[j]
      local bi = {math.max(bb[1],bbgt[1]), math.max(bb[2],bbgt[2]),
                  math.min(bb[3],bbgt[3]), math.min(bb[4],bbgt[4])}
      local iw = bi[3]-bi[1]+1
      local ih = bi[4]-bi[2]+1
      if iw>0 and ih>0 then
        -- compute overlap as area of intersection / area of union
        local ua = (bb[3]-bb[1]+1)*(bb[4]-bb[2]+1)+
                   (bbgt[3]-bbgt[1]+1)*(bbgt[4]-bbgt[2]+1)-iw*ih
        local ov = iw*ih/ua
        if ov > ovmax then
          ovmax = ov
          jmax = j
        end
      end
    end

    local ok = 1
    if jmax ~= -1 and used[jmax] == 0 then
      used[jmax] = 1 -- mark as taken
    else
      ok = 0
    end

    -- record the best box, the overlap, and the fact that we need to score the language match
    local record = {}
    record.ok = ok -- whether this prediction can be counted toward a true positive
    record.ov = ovmax
    if jmax ~= -1 then
      record.candidate = class
    end
    -- Replace nil with empty table to prevent crash in meteor bridge
    --if record.references == nil then record.references = {} end
    record.imgid = self.n
    table.insert(self.records, record)
  end
  
  -- keep track of results
  self.n = self.n + 1
  self.npos = self.npos + nt
  table.insert(self.all_scores, Y:double()) -- inserting the sorted scores as double
end

function DenseCaptioningEvaluator:evaluate(verbose)
  if verbose == nil then verbose = true end
  local min_overlaps = {0.3, 0.4, 0.5, 0.6, 0.7}

  -- concatenate everything across all images
  local scores = torch.cat(self.all_scores, 1) -- concat all scores
  -- call python to evaluate all records and get their BLEU/METEOR scores
  -- local blob = eval_utils.score_labels(self.records) -- replace in place (prev struct will be collected)
  -- local scores = blob.scores -- scores is a list of scores, parallel to records
  collectgarbage()
  collectgarbage()

  -- prints/debugging
  if verbose then
    for k=1,#self.records do
      local record = self.records[k]
      if record.ov > 0 and record.ok == 1 and k % 1000 == 0 then
        print(string.format('IMG %d PRED: %s, OK:%f, OV: %f SCORE: %f',
              record.imgid, record.candidate, record.ok, record.ov, scores[k]))
      end  
    end
  end

  -- lets now do the evaluation
  local y,ix = torch.sort(scores,1,true) -- true makes order descending

  local ap_results = {}
  for foo, min_overlap in pairs(min_overlaps) do
    -- go down the list and build tp,fp arrays
    local n = y:nElement()
    local tp = torch.zeros(n)
    local fp = torch.zeros(n)

    for i=1,n do
      -- pull up the relevant record
      local ii = ix[i]
      local r = self.records[ii]

      if not r.candidate then 
        fp[i] = 1 -- nothing aligned to this predicted box in the ground truth
      else
        -- ok something aligned. Lets check if it aligned enough, and correctly enough
        local score = scores[ii]
        if r.ov >= min_overlap and r.ok == 1 then
          tp[i] = 1
        else
          fp[i] = 1
        end
      end
    end

    fp = torch.cumsum(fp,1)
    tp = torch.cumsum(tp,1)
    local rec = torch.div(tp, self.npos)
    local prec = torch.cdiv(tp, fp + tp)

    -- compute max-interpolated average precision
    local ap = 0
    local apn = 0
    for t=0,1,0.01 do
      local mask = torch.ge(rec, t):double()
      local prec_masked = torch.cmul(prec:double(), mask)
      local p = torch.max(prec_masked)
      ap = ap + p
      apn = apn + 1
    end
    ap = ap / apn

    -- store it
    ap_results['ov' .. min_overlap] = ap
  end

  --local map = utils.average_values(ap_results)
  --local detmap = utils.average_values(det_results)

  -- lets get out of here
  local results = ap_results
  return results
end

function DenseCaptioningEvaluator:numAdded()
  return self.n - 1
end

return eval_utils
