---- UI options ----

SHOW_GRID = false
SHOW_COOR = false
SHOW_TILE_MAP = false
SHOW_OBJECTS = false
SHOW_PROJ = true
SHOW_STILL_ENEMIES = true
SHOW_BANNER = false
SHOW_AI_VISION = false
SHOW_NETWORK = false

---- NN input ----

X1 = 0
X2 = 256
DX = 8
Y1 = 42
Y2 = 222
DY = 12
SIGHT_X = 16
SIGHT_Y = 15
NUM_SNAPSHOTS = 3

----

-- Incoming enemies.
function GetIncomingEnemies()
  local ret = {}
  for i=0,6 do
    local addr = 0x203 + 0x10 * i
    local x = memory.readbyte(addr)
    local y = memory.readbyte(addr + 1)
    if x > 0 and y > 0 then
      addr = 0x718 + i
      local row = memory.readbyte(addr)
      ret[#ret+1] = {x=(x+8)%0xFF, y=y+6, id=i, row=row}
    end
  end
  return ret
end

-- Enemies standing still (encoded).
function GetStillEnemiesEncoded()
  local ret = {}
  ret[#ret+1] = (memory.readbyte(0xE5) + 56) % 256
  for i=0,9 do
    local mask = memory.readbyte(0xC3 + i)
    ret[#ret+1] = mask
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
  local id = 0
  for addr=0x287,0x29F,0x4 do
    local x = memory.readbyte(addr)
    local y = memory.readbyte(addr - 3)
    if x > 0 and y > 0 then
      ret[#ret+1] = {x=x+4, y=y+8, id=id}
    end
    id = id + 1
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

function GetScore()
  local score = 0
  for addr=0x6A0,0x6A5 do
    score = score * 10 + AND(memory.readbyte(addr), 0xF)
  end
  return score
end

function GetSurvivedIncomings(g, p)
  local ret = 0
  if p ~= nil then
    -- This is not accurate, but good enough.
    ret = math.max(0, #p.bullets - #g.bullets) + math.max(0, #p.incoming_enemies - #g.incoming_enemies)
  end
  return ret
end

function GetGame()
  local g = {}
  g.galaxian = {}
  g.galaxian.x = (memory.readbyte(0xE4) + 128) % 256
  g.galaxian.y = Y2-8
  g.still_enemies_encoded = GetStillEnemiesEncoded()
  g.still_enemies = GetStillEnemies()
  g.incoming_enemies = GetIncomingEnemies()
  g.bullets = GetBullets()
  g.missile = GetMissile()
  g.score = GetScore()
  g.lifes = memory.readbyte(0x42)
  g.sight = GetSight(g.galaxian.x)
  return g
end

function IsDead()
  return memory.readbyte(0x41) ~= 0
end

function GetLevel()
  return memory.readbyte(0x59)
end

function ToAction(controls)
  local a = controls['A']
  local l = controls['left']
  local r = controls['right']
  if a then
    if l then return 'l' end
    if r then return 'r' end
    return 'A'
  else
    if l then return 'L' end
    if r then return 'R' end
    return '_'
  end
end

function InSight(sight, t)
  return sight.x1 <= t.x and t.x < sight.x2 and sight.y1 <= t.y and t.y < sight.y2
end

function InSightX(sight, t)
  return sight.x1 <= t.x and t.x < sight.x2
end

function FindBullet(g, id)
  for _, e in pairs(g.bullets) do
    if e.id == id then
      return e
    end
  end
  return nil
end

function FindIncomingEnemies(g, id)
  for _, e in pairs(g.incoming_enemies) do
    if e.id == id then
      return e
    end
  end
  return nil
end

---- UI ----

function IsTileNeuron(id)
  return IsInputNeuron(id) and id:find("%.") ~= nil
end

function Show(recent_games, genome)
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
      gui.drawline(g.galaxian.x, 42, g.galaxian.x, g.galaxian.y, 'red')
    end
    -- galaxian
    gui.drawbox(g.galaxian.x-2, g.galaxian.y-2, g.galaxian.x+2, g.galaxian.y+2, 'green', 'clear')
  end

  if SHOW_TILE_MAP then
    function DrawStillEnemy(e, gid)
      if InSightX(g.sight, e) then
        local x = math.floor((e.x - g.sight.x0) / DX) * DX + g.sight.x0
        local y = Y1+5*DY
        local d = gid
        gui.drawbox(x+d, y+d, x+DX-d, y+DY-d, 'blue', 'clear')
      end
    end
    function DrawTile(t, color, gid)
      if InSight(g.sight, t) then
        local x = math.floor((t.x - g.sight.x0) / DX) * DX + g.sight.x0
        local y = math.floor((t.y - g.sight.y0) / DY) * DY + g.sight.y0
        local d = gid
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

  if SHOW_OBJECTS then
    if SHOW_STILL_ENEMIES then
      for _, e in pairs(g.still_enemies) do
        gui.drawbox(e.x - 4, e.y - 4, e.x + 4, e.y + 4, {0xFF, 0, 0, 0x80}, 'clear')
      end
    end
    for _, e in pairs(g.incoming_enemies) do
      gui.drawbox(e.x - 4, e.y - 4, e.x + 4, e.y + 4, {0xFF, 0, 0, 0x80}, 'clear')
    end
    for _, b in pairs(g.bullets) do
      gui.drawbox(b.x - 2, b.y - 4, b.x + 2, b.y + 4, {0xFF, 0xFF, 0, 0x80}, 'clear')
    end
    if not SHOW_AI_VISION then
      gui.drawbox(g.galaxian.x - 4, g.galaxian.y, g.galaxian.x + 4, g.galaxian.y + 8, 'green')
    end
  end

  if SHOW_PROJ then
    local pg = recent_games[4]
    if pg ~= nil then
      local y = g.galaxian.y
      for _, b in pairs(g.bullets) do
        local pb = FindBullet(pg, b.id)
        if pb ~= nil and pb.y < b.y and b.y < y then
          local x = (b.x-pb.x)/(b.y-pb.y)*(y-pb.y)+pb.x
          --gui.drawline(x, y, b.x, b.y, 'yellow')
          gui.drawbox(x-2, y-2, x+2, y+2, 'yellow')
        end
      end
      for _, e in pairs(g.incoming_enemies) do
        local x = nil
        local pe = FindIncomingEnemies(pg, e.id)
        if pe ~= nil and pe.y < e.y and e.y < y then
          x = (e.x-pe.x)/(e.y-pe.y)*(y-pe.y)+pe.x
        elseif y <= e.y and e.y < y + 12 then
          x = e.x
        end
        if x ~= nil then
          --gui.drawline(x, y, e.x, e.y, 'red')
          gui.drawbox(x-2, y-2, x+2, y+2, 'red')
        end
      end
    end
  end

  if SHOW_NETWORK and genome ~= nil then
    function NeuronPosition(id)
      local y = Y2 - DY
      if id == "o1" then
        return {x=g.galaxian.x-DX, y=y}
      end
      if id == "o2" then
        return {x=g.galaxian.x+DX, y=y}
      end
      if id == "o3" then
        return {x=g.galaxian.x, y=y}
      end
      y = y - DY
      if IsHiddenNeuron(id) then
        local hid = tonumber(id:sub(2))
        if hid % 2 == 0 then
          hid = -hid-1
        end
        return {x=g.galaxian.x + hid * 4, y=y}
      end
      y = y - DY
      if id == "m" then
        return {x=g.galaxian.x-2*DX, y=y}
      end
      if id == "gx" then
        return {x=g.galaxian.x-1*DX, y=y}
      end
      if id == "bias" then
        return {x=g.galaxian.x+1*DX, y=y}
      end
      if id == "rnd" then
        return {x=g.galaxian.x+2*DX, y=y}
      end
      local kind = id:sub(1, 1)
      if kind == 's' then
        local ix = tonumber(id:sub(4, 6))
        local x = g.galaxian.x + ix*DX + DX/2
        local y = Y1+6*DY
        return {x=x, y=y}
      end
      if kind == 'i' or kind == 'b' then
        local ix = tonumber(id:sub(4, 6))
        local iy = tonumber(id:sub(8, 10))
        local x = g.galaxian.x + ix*DX + DX/2
        local y = g.sight.y0 + iy*DY + DY/2
        return {x=x, y=y}
      end
      emu.print("Invalid id:", id)
      return nil
    end
    function DrawNeuron(n)
      if X1 <= n.x and n.x <= X2 then
        gui.drawbox(n.x-2, n.y-2, n.x+2, n.y+2, 'yellow', 'clear')
      end
    end
    for _, gene in pairs(genome.genes) do
      if gene.enabled then
        local src = NeuronPosition(gene.src)
        local out = NeuronPosition(gene.out)
        if gene.weight ~= 0 then
          local color
          if gene.weight > 0 then
            color = {0, 0xFF, 0, 0xFF * gene.weight}
          else
            color = {0xFF, 0, 0, 0xFF * -gene.weight}
          end
          gui.drawline(src.x, src.y, out.x, out.y, color)
        end
        if not IsTileNeuron(gene.src) then
          DrawNeuron(src)
        end
        if not IsTileNeuron(gene.out) then
          DrawNeuron(out)
        end
      end
    end
    for _, id in pairs({"m", "gx", "bias", "rnd"}) do
      DrawNeuron(NeuronPosition(id))
    end
    for o=1,NUM_OUTPUT_NEURONS do
      DrawNeuron(NeuronPosition("o"..o))
    end
  end

  if SHOW_BANNER and genome ~= nil then
    local measured = 0
    local total = 0
    for _,species in pairs(pool.species) do
      for _,genome in pairs(species.genomes) do
        total = total + 1
        if genome.fitness ~= 0 then
          measured = measured + 1
        end
      end
    end
    gui.drawbox(0, 0, 300, 26, {0xFF, 0xFF, 0xFF, '0x80'}, 'clear')
    gui.drawtext(5, 10, "Gen " .. pool.generation .. " species " ..
        pool.cur_species .. " genome " .. pool.cur_genome ..
        " (" .. math.floor(measured/total*100) .. "%)", 'black', 'clear')
    gui.drawtext(5, 18, "Fitness: " .. genome.fitness, 'black', 'clear')
    gui.drawtext(100, 18, "Max Fitness: " .. pool.max_fitness, 'black', 'clear')
  end
end

---- misc ----

function SkipFrames(frames)
  for i=1,frames do
    emu.frameadvance();
  end
end

function Reset()
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
