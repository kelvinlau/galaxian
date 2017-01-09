-- Host a server to allow python to control and read the game through files.
--
-- Author: kelvinlau

require("game")

---- Configs ----

INPUT = "Z:/Users/kelvinlau/neat/galaxian.ctrl"
OUTPUT = "Z:/Users/kelvinlau/neat/galaxian.frames"
RESET = true

---- Responding ----

function Respond(client, data)
  --emu.print("Respond", data.seq)
  client:send(data.seq .. "\n")
  client:send(data.control .. "\n")
  client:send(data.reward .. "\n")
  for id, val in pairs(data.inputs) do
    client:send(id .. "\n" .. val .. "\n")
  end
  client:send("done\n")
  --emu.print("Responded", data.seq)
end

---- Control reading ----

function Split(s, delimiter)
  local result = {}
  for match in (s..delimiter):gmatch("(.-)"..delimiter) do
    table.insert(result, match)
  end
  return result[1], result[2]
end

function ReadControl(client)
  local line, err = client:receive()
  if err ~= nil then
    emu.print(err)
    emu.message(err)
    return
  end

  --emu.print(line)
  --emu.message(line)

  local action = nil
  local seq = nil
  action, seq = Split(line, ' ')

  ctrl = {}
  if action ~= 'human' then
    ctrl['A'] = false
    ctrl['left'] = false
    ctrl['right'] = false
    if action == 'fire' then
      ctrl['A'] = true
    elseif action == 'left' then
      ctrl['left'] = true
    elseif action == 'right' then
      ctrl['right'] = true
    end
  else
    emu.speedmode('normal')
  end

  return ctrl, seq
end

---- UI ----

function ShowScore(max_score)
  gui.drawtext(10, 10, "Max Score " .. max_score)
end

---- Script starts here ----

emu.print("Running Galaxian server")

if RESET then
  Reset()
end

emu.speedmode("normal")

INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE)

local recent_games = {}
local prev_g = nil
local prev_score = 0
local max_score = 0
local cur_step = 0

local socket = require("socket")
local server = assert(socket.bind("*", 62345))
local ip, port = server:getsockname()
emu.print("localhost:" .. port)
emu.message("localhost:" .. port)
SkipFrames(60)
local client = server:accept()
-- client:settimeout(1)
-- client:close()

while true do
  local control = nil
  local seq = nil
  control, seq = ReadControl(client)

  cur_step = cur_step + 1
  for i = 1, 20 do
    ShowScore(max_score)
    joypad.set(1, control)
    emu.frameadvance();
  end

  local g = GetGame()
  recent_games[0] = g

  local data = {}
  data.seq = seq
  data.inputs = GetInputs(recent_games, true)  -- exclude_bias=true
  data.control = GetControl()
  data.reward = g.score - prev_score + GetSurvivedIncomings(g, prev_g) * 100
  Respond(client, data)

  prev_g = g;
  prev_score = g.score
  max_score = math.max(max_score, g.score)

  -- Add a snapshot for every 6 steps (30 frames).
  if cur_step % 6 == 0 then
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
    cur_step = 0
    prev_score = 1000  -- Next respond with have reward = -1000.
  end
end
