-- Host a server to allow python to control and read the game through socket.
--
-- Author: kelvinlau

require("auxlib");
require("game")

SHOW_AI_VISION = false
SHOW_OBJECTS = false
SHOW_STILL_ENEMIES = false

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
    Append(e.id)
    Append(e.x)
    Append(e.y)
  end
  Append(#g.bullets)
  for _, e in pairs(g.bullets) do
    Append(e.id)
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
    elseif action == 'l' then
      ctrl['left'] = true
      ctrl['A'] = true
    elseif action == 'r' then
      ctrl['right'] = true
      ctrl['A'] = true
    end
  end

  return ctrl, seq
end

---- UI ----

function ShowScore(score, max_score)
  gui.drawtext(10, 10, "Score " .. score)
  gui.drawtext(100, 10, "Max Score " .. max_score)
end

---- Small mode ----

function Hit(rg)
  if #rg[1].incoming_enemies >  0 then return false end
  if #rg[2].incoming_enemies >  0 then return false end
  if #rg[3].incoming_enemies == 0 then return false end
  if #rg[4].incoming_enemies == 0 then return false end
  for i = 1, 3 do
    if #rg[i].still_enemies ~= #rg[4].still_enemies then
      return false
    end
  end
  return true
end

---- Script starts here ----

emu.print("Running Galaxian server")

Reset()
INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE)
RELOAD_STATE = savestate.create(8)
savestate.save(RELOAD_STATE)

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
    title="Show AI vision",
    action=
      function (self)
        SHOW_AI_VISION = not SHOW_AI_VISION
        SHOW_OBJECTS = not SHOW_OBJECTS
      end
  },
  iup.button{
    title="Save",
    action=
      function (self)
        savestate.save(INIT_STATE)
        savestate.save(RELOAD_STATE)
      end
  },
  iup.button{
    title="Reset",
    action=
      function (self)
        Reset()
        savestate.save(INIT_STATE)
        savestate.save(RELOAD_STATE)
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

local recent_games = {}
local reward_sum = 0

while true do
  if math.random() < 0.01 then
    savestate.save(RELOAD_STATE)
  end

  local control = nil
  local seq = nil
  control, seq = ReadControl(client)

  local reward = 0
  local terminal = false

  -- Advance 5 frames. If dead, start over.
  for i = 1, 5 do
    recent_games[0] = GetGame()
    Show(recent_games)
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
    g = GetGame()
    if IsDead() then
      reward = -1
      terminal = true
      break
    end
  end

  local g = GetGame()
  if #recent_games == 0 then
    for i = 1, 4 do
      recent_games[i] = g
    end
  else
    for i = 4, 2, -1 do
      recent_games[i] = recent_games[i-1]
    end
    recent_games[1] = g
  end
  recent_games[0] = g
  local action = ToAction(control)
  if not terminal then
    reward = g.score - recent_games[2].score
  end
  Respond(client, seq, g, action, reward, terminal)

  reward_sum = reward_sum + reward
  max_score = math.max(max_score, reward_sum)

  if terminal then
    emu.print("Score: " .. g.score)
    if math.random() < 0.1 then
      savestate.load(INIT_STATE)
    else
      savestate.load(RELOAD_STATE)
    end
    reward_sum = 0
    recent_games = {}
  end
end
