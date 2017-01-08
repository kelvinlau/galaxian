-- Records human playing and save neural network inputs as training data.
--
-- Author: kelvinlau

require("game")

---- Configs ----

INPUT = "Z:/Users/kelvinlau/neat/galaxian.ctrl"
OUTPUT = "Z:/Users/kelvinlau/neat/galaxian.frames"
RESET = true

---- Saving ----

function Save(data)
  local file = io.open(OUTPUT, "a")
  file:write(data.seq .. " " .. data.control .. "\n")
  for id, val in pairs(data.inputs) do
    file:write(id .. "\n" .. val .. "\n")
  end
  file:write("done\n")
  file:close()
end

---- Control reading ----

INPUT_FILE = nil

function split(s, delimiter)
  local result = {};
  for match in (s..delimiter):gmatch("(.-)"..delimiter) do
    table.insert(result, match);
  end
  return result;
end

function ReadControls()
  if INPUT_FILE == nil then
    INPUT_FILE = io.open(INPUT, "r")
  end

  line = INPUT_FILE:read("*line")
  if line == nil then
    line = 'no ctrl'
  end
  emu.message(line)

  ctrl = {}
  ctrl['A'] = false
  ctrl['left'] = false
  ctrl['right'] = false
  ctrl_name, seq = split(line, ' ')
  if ctrl_name == 'fire' then
    ctrl['A'] = true
  elseif ctrl_name == 'left' then
    ctrl['left'] = true
  elseif ctrl_name == 'right' then
    ctrl['right'] = true
  end
  return ctrl, seq
end

---- Script starts here ----

emu.print("Running Galaxian server")

if RESET then
  Reset()
end

emu.speedmode("normal")

INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE)

local controls = {}
local recent_games = {}
local cur_frame = 0

while true do
  -- Read buttons for every 5 frames.
  local seq = nil
  if cur_frame % 5 == 0 then
    controls, seq = ReadControls()
  end
  joypad.set(1, controls)

  cur_frame = cur_frame + 1

  emu.frameadvance();

  local g = GetGame()
  recent_games[0] = g

  Show(recent_games)

  if seq ~= nil then
    local data = {}
    data.seq = seq
    data.inputs = GetInputs(recent_games)
    data.control = GetControl()
    Save(data)
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

  -- Reset if dead.
  if g.lifes < 2 then
    savestate.load(INIT_STATE)
    recent_games = {}
    cur_frame = 0
  end
end
