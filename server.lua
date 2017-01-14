-- Host a server to allow python to control and read the game through socket.
--
-- Author: kelvinlau

require("auxlib");
require("game")

---- Responding ----

function Respond(client, seq, g, action, reward, terminal)
  --emu.print("Respond", seq)
  local line = ""
  function Append(x)
    line = line .. " " .. x
  end

  Append(seq)
  Append(reward)
  if terminal then
    Append(1)
  else
    Append(0)
  end
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

function ShowScore(score, max_score)
  gui.drawtext(10, 10, "Score " .. score)
  gui.drawtext(60, 10, "Max Score " .. max_score)
end

---- Script starts here ----

emu.print("Running Galaxian server")

Reset()
INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE)

human_play = false
max_score = 0

-- Dialog.
dialogs = dialogs + 1;
handles[dialogs] = iup.dialog{iup.vbox{
  iup.button{
    title="Human play",
    action=
      function (self)
        human_play = not human_play
        if human_play then
          emu.speedmode("normal")
        else
          emu.speedmode("turbo")
        end
      end
  },
  iup.button{
    title="Clear score",
    action=
      function (self)
        max_score = 0
      end
  },
  iup.button{
    title="Save",
    action=
      function (self)
        savestate.save(INIT_STATE)
      end
  },
  iup.button{
    title="Reset",
    action=
      function (self)
        Reset()
        savestate.save(INIT_STATE)
      end
  },
  margin="20x20"},
  title=""}
handles[dialogs]:show();

-- Socket.
local socket = require("socket")
local server = assert(socket.bind("*", 62343))
local ip, port = server:getsockname()
emu.print("localhost:" .. port)
emu.message("localhost:" .. port)
SkipFrames(60)
local client = server:accept()
-- client:settimeout(1)
-- client:close()

SMALL_MODE = true

local prev_score = 0
local prev_inum = 0
local prev_snum = 0
local reward_sum = 0

while true do
  local control = nil
  local seq = nil
  control, seq = ReadControl(client)

  local reward = 0
  local terminal = false

  -- Advance 5 frames. If dead, start over.
  for i = 1, 5 do
    ShowScore(reward_sum, max_score)
    if human_play and i == 1 then
      joypad.set(1, {})
    else
      joypad.set(1, control)
    end
    emu.frameadvance()
    if human_play and i == 1 then
      control = joypad.get(1)
    end
    local g = GetGame()
    if IsDead() then
      reward = 0
      terminal = true
      break
    elseif #g.incoming_enemies > 1 then
      reward = 1
      terminal = true
      break
    end
  end

  local g = GetGame()
  local inum = #g.incoming_enemies
  local snum = #g.still_enemies
  local action = ToAction(control)
  if not terminal then
    if SMALL_MODE then
      if inum < prev_inum and snum == prev_snum then
        reward = 1
      end
    else
      reward = g.score - prev_score
    end
  end
  Respond(client, seq, g, action, reward, terminal)

  if terminal then
    SkipFrames(60)  -- Show explosion.
    savestate.load(INIT_STATE)
    reward_sum = 0
    prev_score = 0
    prev_inum = 0
    prev_snum = 0
  else
    prev_score = g.score
    prev_inum = inum
    prev_snum = snum
  end

  reward_sum = reward_sum + reward
  max_score = math.max(max_score, reward_sum)
end
