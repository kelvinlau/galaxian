-- Galaxian using Neural Network and Genetic Algorithm
-- Author: kelvinlau
--
-- Records human playing and save neural network inputs as training data.

---- Configs ----

FILENAME = "Z:/Users/kelvinlau/neat/galaxian.in"
RESET = true

SHOW_GRID = false
SHOW_COOR = false
SHOW_TILE_MAP = true
SHOW_AI_VISION = true

---- NN input ----

X1 = 0
X2 = 256
DX = 4
Y1 = 42
Y2 = 222
DY = 6
SIGHT_X = 48
SIGHT_Y = 32
NUM_SNAPSHOTS = 3

----

function InSight(sight, t)
  return sight.x1 <= t.x and t.x < sight.x2 and sight.y1 <= t.y and t.y < sight.y2
end

function InSightX(sight, t)
  return sight.x1 <= t.x and t.x < sight.x2
end

function GetInputs(recent_games)
  local inputs = {}
  local g = recent_games[0]

  function AddVal(id, val)
    if inputs[id] == nil then
      inputs[id] = val
    else
      inputs[id] = inputs[id] + val
    end
  end

  function AddStillEnemy(prefix, e, val)
    if InSightX(g.sight, e) then
      local ix = math.floor((e.x - g.galaxian_x) / DX)
      local id = string.format("%s.%03d", prefix, ix)
      AddVal(id, val)
    end
  end

  function AddTile(prefix, t, val)
    if InSight(g.sight, t) then
      local ix = math.floor((t.x - g.galaxian_x) / DX)
      local iy = math.floor((t.y - g.sight.y0) / DY)
      local id = string.format("%s.%03d,%03d", prefix, ix, iy)
      AddVal(id, val)
    end
  end

  -- Tiles (last several snapshots)
  -- TODO: enemy type?
  for gid, game in pairs(recent_games) do
    for _, e in pairs(game.still_enemies) do
      AddStillEnemy("s" .. gid, e, 1)
    end
    for _, e in pairs(game.incoming_enemies) do
      AddTile("i" .. gid, e, 1)
    end
    for _, b in pairs(game.bullets) do
      AddTile("b" .. gid, b, 1)
    end
  end

  -- misile_y scaled in [0, 1]
  inputs["m"] = g.missile.y / 200

  -- galaxian_x scaled in [0, 1]
  inputs["gx"] = (g.galaxian_x - X1) / (X2 - X1)

  if DEBUG then
    for id, _ in pairs(inputs) do
      if not LinearSearch(ALL_INPUTS, id) then
        emu.print("Unexpected input:", id)
      end
    end
  end

  return inputs
end

function GetOutputs()
  local outputs = joypad.get(1)
  local ret = {}
  ret["A"] = outputs["A"]
  ret["left"] = outputs["left"]
  ret["right"] = outputs["right"]
  return ret
end

---- Game ----

-- Incoming enemies.
function GetIncomingEnemies()
  local ret = {}
  for addr=0x203,0x253,0x10 do
    local x = memory.readbyte(addr)
    local y = memory.readbyte(addr + 1)
    if x > 0 and y > 0 then
      ret[#ret+1] = {x=(x+8)%0xFF, y=y+6}
    end
  end
  return ret
end

-- Enemies standing still.
function GetStillEnemies()
  local ret = {}
  local dx = memory.readbyte(0xE5)
  for i=0,9 do
    local x = (dx + 48 + 16 * i + 8) % 0xFF
    local y = 108
    local mask = memory.readbyte(0xC3 + i)
    while mask > 0 do
      if mask % 2 ~= 0 then
        ret[#ret+1] = {x=x, y=y}
        mask = mask - 1
      end
      mask = mask / 2
      y = y - 12
    end
  end
  return ret
end

-- Incoming enemy bullets.
function GetBullets()
  local ret = {}
  for addr=0x28B,0x29F,0x4 do
    local x = memory.readbyte(addr)
    local y = memory.readbyte(addr - 3)
    if x > 0 and y > 0 then
      ret[#ret+1] = {x=x+4, y=y+8}
    end
  end
  return ret
end

-- Sight rectangle.
function GetSight(galaxian_x)
  local sight = {
    x0 = galaxian_x - SIGHT_X * DX,
    y0 = Y2 - SIGHT_Y * DY,
  }
  sight.x1 = galaxian_x - math.min(math.floor((galaxian_x-X1)/DX), SIGHT_X) * DX
  sight.x2 = galaxian_x + math.min(math.floor((X2-galaxian_x)/DX), SIGHT_X) * DX
  sight.y1 = sight.y0
  sight.y2 = Y2
  return sight
end

function Round(x, x1, dx)
  return (x-x1) - (x-x1)%dx + x1
end

-- Our missile. nil if not fired.
function GetMissile()
  local x = memory.readbyte(0x283)
  local y = memory.readbyte(0x280)
  return {x=x, y=y}
end

---- File ----

function Save(frame_data, end_frame)
  emu.print("Saving frames up to " .. end_frame)
  local file = io.open(FILENAME, "a")
  for frame, data in pairs(frame_data) do
    if frame < end_frame then
      for id, val in pairs(data.inputs) do
        file:write(id .. "\n" .. val .. "\n")
      end
      file:write("done\n")
      for button, val in pairs(data.outputs) do
        if val then
          file:write(button .. "\n")
        end
      end
      file:write("done\n")
      frame_data[frame] = nil
    end
  end
  file:close()
end

---- UI ----

function IsTileNeuron(id)
  return IsInputNeuron(id) and id:find("%.") ~= nil
end

function Show(recent_games)
  local g = recent_games[0]

  if SHOW_AI_VISION then
    gui.drawbox(0, 0, 256, 256, 'black', 'clear')
  end

  if SHOW_GRID then
    local color = {0xFF, 0xFF, 0xFF, 0x80}
    for x = g.sight.x1, g.sight.x2, DX do
      gui.drawline(x, g.sight.y1, x, g.sight.y2, color)
      if SHOW_COOR then
        local y = 165 + (x % (3 * DX) / DX) * DY
        gui.drawtext(x, y, x)
      end
    end
    for y = g.sight.y1, g.sight.y2, DY do
      gui.drawline(g.sight.x1, y, g.sight.x2, y, color)
      if SHOW_COOR then
        gui.drawtext(5, y, y)
      end
    end
  end

  if SHOW_AI_VISION then
    -- missile aimming ray
    if g.missile.y == 200 then
      gui.drawline(g.galaxian_x, g.sight.y1, g.galaxian_x, g.sight.y2, 'red')
    end
    gui.drawbox(g.galaxian_x-2, g.galaxian_y-2, g.galaxian_x+2, g.galaxian_y+2, 'green', 'clear')
  end

  if SHOW_TILE_MAP then
    function DrawStillEnemy(e, gid)
      if InSightX(g.sight, e) then
        local x = math.floor((e.x - g.sight.x0) / DX) * DX + g.sight.x0
        local y = Y1+5*DY
        local d = math.min(gid, DX/2-1)
        gui.drawbox(x+d, y+d, x+DX-d, y+DY-d, 'blue', 'clear')
      end
    end
    function DrawTile(t, color, gid)
      if InSight(g.sight, t) then
        local x = math.floor((t.x - g.sight.x0) / DX) * DX + g.sight.x0
        local y = math.floor((t.y - g.sight.y0) / DY) * DY + g.sight.y0
        local d = math.min(gid, DX/2-1)
        gui.drawbox(x+d, y+d, x+DX-d, y+DY-d, color, 'clear')
      end
    end
    for gid = NUM_SNAPSHOTS-1, 0, -1 do
      local game = recent_games[gid]
      if game ~= nil then
        for _, e in pairs(game.still_enemies) do
          DrawStillEnemy(e, gid)
        end
        for _, e in pairs(game.incoming_enemies) do
          DrawTile(e, 'blue', gid)
        end
        for _, b in pairs(game.bullets) do
          DrawTile(b, 'red', gid)
        end
      end
    end
  end
end

---- Misc ----

function SkipFrames(frames)
  for i=1,frames do
    emu.frameadvance();
  end
end

---- Script starts here ----

emu.print("Running Galaxian recorder")

if RESET then
  emu.softreset()
  emu.speedmode("turbo")
  for k=1,2 do
    SkipFrames(60)
    for i=1,10 do
      joypad.set(1, {start=true})
      emu.frameadvance();
    end
    for i=1,10 do
      joypad.set(1, {start=false})
      emu.frameadvance();
    end
  end
  SkipFrames(240)
end

emu.speedmode("normal")

INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE);

local recent_games = {}
local cur_frame = 0
local frame_data = {}

while true do
  local g = {}
  g.galaxian_x = (memory.readbyte(0xE4) + 128) % 256
  g.galaxian_y = Y2
  g.still_enemies = GetStillEnemies()
  g.incoming_enemies = GetIncomingEnemies()
  g.bullets = GetBullets()
  g.missile = GetMissile()
  g.lifes = memory.readbyte(0x42)
  g.sight = GetSight(g.galaxian_x)

  recent_games[0] = g

  -- Record input and output for every 5 frames.
  if cur_frame % 5 == 0 then
    local data = {}
    data.inputs = GetInputs(recent_games)
    data.outputs = GetOutputs()
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
