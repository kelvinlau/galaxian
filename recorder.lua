-- Galaxian using Neural Network and Genetic Algorithm
-- Author: kelvinlau
--
-- Records human playing and save neural network inputs as training data.

---- Configs ----

FILENAME = "Z:/Users/kelvinlau/neat/galaxian.frames"
RESET = true

SHOW_TILE_MAP = true
SHOW_AI_VISION = true

---- Saving ----

function Save(frame_data, end_frame)
  emu.print("Saving frames up to " .. end_frame)
  local file = io.open(FILENAME, "a")
  for frame, data in pairs(frame_data) do
    if frame < end_frame then
      file:write(data.control .. "\n")
      for id, val in pairs(data.inputs) do
        file:write(id .. "\n" .. val .. "\n")
      end
      file:write("done\n")
      frame_data[frame] = nil
    end
  end
  file:close()
end

---- Script starts here ----

emu.print("Running Galaxian recorder")

if RESET then
  Reset()
end

emu.speedmode("normal")

INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE);

local recent_games = {}
local cur_frame = 0
local frame_data = {}

while true do
  local g = GetGame()

  recent_games[0] = g

  -- Record input and output for every 5 frames.
  if cur_frame % 5 == 0 then
    local data = {}
    data.inputs = GetInputs(recent_games)
    data.control = GetControl()
    frame_data[cur_frame] = data
  end

  -- Save to file for every 600 frames, save up to 300 frames ago.
  if cur_frame % 600 == 300 then
    Save(frame_data, cur_frame-300)
  end

  -- Add a snapshot for every 30 frames.
  if cur_frame % 30 == 0 then
    for i = NUM_SNAPSHOTS-1,2,-1 do
      if recent_games[i-1] ~= nil then
        recent_games[i] = recent_games[i-1]
      end
    end
    recent_games[1] = g
  end

  cur_frame = cur_frame + 1

  Show(recent_games)

  -- Reset if dead.
  if g.lifes < 2 then
    savestate.load(INIT_STATE)
    recent_games = {}
    cur_frame = 0
    frame_data = {}
  end

  emu.frameadvance();
end
