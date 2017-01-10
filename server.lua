-- Host a server to allow python to control and read the game through socket.
--
-- Author: kelvinlau

require("game")

---- Configs ----

RESET = true

---- Responding ----

function Respond(client, seq, g, action, reward)
  --emu.print("Respond", seq)
  local line = ""
  function Append(x)
    line = line .. " " .. x
  end

  Append(seq)
  Append(reward)
  Append(action)
  Append(g.galaxian.x)
  Append(g.galaxian.y)
  Append(g.missile.x)
  Append(g.missile.y)
  for _, e in pairs(g.still_enemies_encoded) do
    Append(e)
  end
  Append(#g.incoming_enemies)
  for _, e in pairs(g.incoming_enemies) do
    Append(e.x)
    Append(e.y)
  end
  Append(#g.bullets)
  for _, e in pairs(g.bullets) do
    Append(e.x)
    Append(e.y)
  end

  client:send(line .. "\n")
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
  if action ~= 'H' then
    ctrl['A'] = false
    ctrl['left'] = false
    ctrl['right'] = false
    if action == 'A' then
      ctrl['A'] = true
    elseif action == 'L' then
      ctrl['left'] = true
    elseif action == 'R' then
      ctrl['right'] = true
    end
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
  emu.speedmode("normal")
end

INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE)

local prev_score = 0
local max_score = 0

local socket = require("socket")
local server = assert(socket.bind("*", 62343))
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

  for i = 1, 12 do
    ShowScore(max_score)
    joypad.set(1, control)
    emu.frameadvance();
  end

  local g = GetGame()
  local action = GetAction()
  local reward = g.score - prev_score
  Respond(client, seq, g, action, reward)

  prev_score = g.score
  max_score = math.max(max_score, g.score)

  -- Reset if dead.
  if g.lifes < 2 then
    savestate.load(INIT_STATE)
    prev_score = 1000  -- Next respond with have reward = -1000.
  end
end
