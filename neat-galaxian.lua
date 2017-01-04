function DrawEnemy(x_addr, y_addr)
  local x = memory.readbyte(x_addr)
  local y = memory.readbyte(y_addr)
  if x > 0 and y > 0 then
    gui.drawbox(x, y, x + 16, y + 16, {0xFF, 0, 0, 0x80}, 'clear')
  end
end

function DrawBullet(x_addr, y_addr)
  local x = memory.readbyte(x_addr)
  local y = memory.readbyte(y_addr)
  if x > 0 and y > 0 then
    gui.drawbox(x, y, x + 8, y + 8, {0xFF, 0xFF, 0, 0x80}, 'clear')
  end
end

function GetScore()
  local score = 0
  for addr=0x6A0,0x6A5 do
    score = score * 10 + AND(memory.readbyte(addr), 0xF)
  end
  return score
end

DRAW_COORDINATES = false

while true do
  if DRAW_COORDINATES then
    for x=20,0xF0,20 do
      gui.drawline(x, 140, x, 160, 'white')
      gui.drawtext(x, 165, x)
    end

    for y=12,220,12 do
      gui.drawline(15, y, 50, y, 'white')
      gui.drawtext(5, y, y)
    end
  end

  self_x = (memory.readbyte(0xE4) + 124) % 256
  self_y = 200
  gui.drawbox(self_x, self_y, self_x + 8, self_y + 8, 'green')

  DrawEnemy(0x203, 0x204)
  DrawEnemy(0x213, 0x214)
  DrawEnemy(0x223, 0x224)
  DrawEnemy(0x233, 0x234)
  DrawEnemy(0x243, 0x244)
  DrawEnemy(0x253, 0x254)

  DrawBullet(0x29F, 0x29C)
  DrawBullet(0x29B, 0x298)
  DrawBullet(0x297, 0x294)
  DrawBullet(0x293, 0x290)
  DrawBullet(0x28F, 0x28C)
  DrawBullet(0x28B, 0x288)

  DrawBullet(0x283, 0x280)

  score = GetScore()
  lifes = memory.readbyte(0x42)

  for ey=48,108,12 do
    for ex=20,240,4 do
      r, g, b, p = emu.getscreenpixel(ex, ey, true)
      gray_scale = math.max(r, g, b)
      gui.drawbox(ex-4, ey-4, ex+4, ey+4, {0xFF, 0, 0, gray_scale}, 'clear')
    end
  end

  emu.frameadvance();
end
