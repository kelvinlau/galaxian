-- Galaxian using Neural Network and Genetic Algorithm
-- Author: kelvinlau
--
-- Inspired by MarI/O by SethBling.
-- Intended for use on the FCEUX emulator.
-- Enter a level, then load this script.

---- Game constants ----

BUTTONS = {"A", "left", "right"}

X1 = 16
X2 = 240
DX = 8
Y1 = 42
Y2 = 222
DY = 12

SIGHT_RADIUS = 10

---- NN and GA constants ----

NUM_SNAPSHOTS = 1
NUM_TILE_MAP_NODES = (SIGHT_RADIUS*2) * SIGHT_RADIUS * NUM_SNAPSHOTS
Inputs = NUM_TILE_MAP_NODES + 3
Outputs = #BUTTONS

Population = 300
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

MaxNodes = 1000000

---- Pool file constants ----

WORK_DIR = "Z:/Users/kelvinlau/neat"
FILENAME = "galaxian.pool"

----

function getInputs(g)
  local inputs = {}
  -- tile_map
  -- TODO: last several snapshots
  -- TODO: enemy type?
  for i = 1,NUM_TILE_MAP_NODES do
    inputs[i] = 0
  end
  for x, row in pairs(g.tile_map) do
    for y, val in pairs(row) do
      local iy = (y - g.sight.y0) / DY
      local ix = (x - g.sight.x0) / DX
      local i = iy * (2 * SIGHT_RADIUS) + ix + 1
      if 1 <= i and i <= NUM_TILE_MAP_NODES then
        inputs[i] = val
      else
        emu.print("Invalid tile:", x, y, val, ix, iy, i)
      end
    end
  end
  -- has misile or not
  -- TODO: maybe misile_y?
  if g.missile ~= nil then
    inputs[#inputs+1] = 1
  else
    inputs[#inputs+1] = 0
  end
  -- galaxian_x scaled in [0, 1]
  inputs[#inputs+1] = (g.galaxian_x - X1) / (X2 - X1)
  -- galaxian_x % DX
  inputs[#inputs+1] = g.galaxian_x % DX / DX
  return inputs
end

function sigmoid(x)
  return 2/(1+math.exp(-4.9*x))-1
end

function newInnovation()
  pool.innovation = pool.innovation + 1
  return pool.innovation
end

function newPool()
  local pool = {}
  pool.species = {}
  pool.generation = 0
  pool.innovation = Outputs
  pool.currentSpecies = 1
  pool.currentGenome = 1
  pool.currentFrame = 0
  pool.maxFitness = 0
  
  return pool
end

function newSpecies()
  local species = {}
  species.topFitness = 0
  species.staleness = 0
  species.genomes = {}
  species.averageFitness = 0
  
  return species
end

function newGenome()
  local genome = {}
  genome.genes = {}
  genome.fitness = 0
  genome.adjustedFitness = 0
  genome.maxneuron = 0
  genome.globalRank = 0
  genome.mutationRates = {}
  genome.mutationRates["connections"] = MutateConnectionsChance
  genome.mutationRates["link"] = LinkMutationChance
  genome.mutationRates["bias"] = BiasMutationChance
  genome.mutationRates["node"] = NodeMutationChance
  genome.mutationRates["enable"] = EnableMutationChance
  genome.mutationRates["disable"] = DisableMutationChance
  genome.mutationRates["step"] = StepSize
  
  return genome
end

function copyGenome(genome)
  local genome2 = newGenome()
  for g=1,#genome.genes do
    table.insert(genome2.genes, copyGene(genome.genes[g]))
  end
  genome2.maxneuron = genome.maxneuron
  genome2.mutationRates["connections"] = genome.mutationRates["connections"]
  genome2.mutationRates["link"] = genome.mutationRates["link"]
  genome2.mutationRates["bias"] = genome.mutationRates["bias"]
  genome2.mutationRates["node"] = genome.mutationRates["node"]
  genome2.mutationRates["enable"] = genome.mutationRates["enable"]
  genome2.mutationRates["disable"] = genome.mutationRates["disable"]
  
  return genome2
end

function basicGenome()
  local genome = newGenome()
  local innovation = 1

  genome.maxneuron = Inputs
  mutate(genome)
  
  return genome
end

function newGene()
  local gene = {}
  gene.into = 0
  gene.out = 0
  gene.weight = 0.0
  gene.enabled = true
  gene.innovation = 0
  
  return gene
end

function copyGene(gene)
  local gene2 = newGene()
  gene2.into = gene.into
  gene2.out = gene.out
  gene2.weight = gene.weight
  gene2.enabled = gene.enabled
  gene2.innovation = gene.innovation
  
  return gene2
end

function newNeuron()
  local neuron = {}
  neuron.incoming = {}
  neuron.value = 0.0
  
  return neuron
end

function generateNetwork(genome)
  local network = {}
  network.neurons = {}
  
  for i=1,Inputs do
    network.neurons[i] = newNeuron()
  end
  
  for o=1,Outputs do
    network.neurons[MaxNodes+o] = newNeuron()
  end
  
  table.sort(genome.genes, function (a,b)
    return (a.out < b.out)
  end)
  for i=1,#genome.genes do
    local gene = genome.genes[i]
    if gene.enabled then
      if network.neurons[gene.out] == nil then
        network.neurons[gene.out] = newNeuron()
      end
      local neuron = network.neurons[gene.out]
      table.insert(neuron.incoming, gene)
      if network.neurons[gene.into] == nil then
        network.neurons[gene.into] = newNeuron()
      end
    end
  end
  
  genome.network = network
end

function evaluateNetwork(network, inputs)
  if #inputs ~= Inputs then
    emu.print("Incorrect number of neural network inputs.", #inputs, Inputs)
    return {}
  end
  
  for i=1,Inputs do
    network.neurons[i].value = inputs[i]
  end
  
  for _,neuron in pairs(network.neurons) do
    local sum = 0
    for j = 1,#neuron.incoming do
      local incoming = neuron.incoming[j]
      local other = network.neurons[incoming.into]
      sum = sum + incoming.weight * other.value
    end
    
    if #neuron.incoming > 0 then
      neuron.value = sigmoid(sum)
    end
  end
  
  local outputs = {}
  for o=1,Outputs do
    local button = BUTTONS[o]
    if network.neurons[MaxNodes+o].value > 0 then
      outputs[button] = true
    else
      outputs[button] = false
    end
  end
  
  -- TODO: Try only 2 outputs: {"direction", "fire"}
  if outputs["left"] and outputs["right"] then
    outputs["left"] = false
    outputs["right"] = false
  end
  
  return outputs
end

function crossover(g1, g2)
  -- Make sure g1 is the higher fitness genome
  if g2.fitness > g1.fitness then
    tempg = g1
    g1 = g2
    g2 = tempg
  end

  local child = newGenome()
  
  local innovations2 = {}
  for i=1,#g2.genes do
    local gene = g2.genes[i]
    innovations2[gene.innovation] = gene
  end
  
  for i=1,#g1.genes do
    local gene1 = g1.genes[i]
    local gene2 = innovations2[gene1.innovation]
    if gene2 ~= nil and math.random(2) == 1 and gene2.enabled then
      table.insert(child.genes, copyGene(gene2))
    else
      table.insert(child.genes, copyGene(gene1))
    end
  end
  
  child.maxneuron = math.max(g1.maxneuron,g2.maxneuron)
  
  for mutation,rate in pairs(g1.mutationRates) do
    child.mutationRates[mutation] = rate
  end
  
  return child
end

function randomNeuron(genes, nonInput)
  local neurons = {}
  if not nonInput then
    for i=1,Inputs do
      neurons[i] = true
    end
  end
  for o=1,Outputs do
    neurons[MaxNodes+o] = true
  end
  for i=1,#genes do
    if (not nonInput) or genes[i].into > Inputs then
      neurons[genes[i].into] = true
    end
    if (not nonInput) or genes[i].out > Inputs then
      neurons[genes[i].out] = true
    end
  end

  local count = 0
  for _,_ in pairs(neurons) do
    count = count + 1
  end
  local n = math.random(1, count)
  
  for k,v in pairs(neurons) do
    n = n-1
    if n == 0 then
      return k
    end
  end
  
  return 0
end

function containsLink(genes, link)
  for i=1,#genes do
    local gene = genes[i]
    if gene.into == link.into and gene.out == link.out then
      return true
    end
  end
end

function pointMutate(genome)
  local step = genome.mutationRates["step"]
  
  for i=1,#genome.genes do
    local gene = genome.genes[i]
    if math.random() < PerturbChance then
      gene.weight = gene.weight + math.random() * step*2 - step
    else
      gene.weight = math.random()*4-2
    end
  end
end

function linkMutate(genome, forceBias)
  local neuron1 = randomNeuron(genome.genes, false)
  local neuron2 = randomNeuron(genome.genes, true)
   
  local newLink = newGene()
  if neuron1 <= Inputs and neuron2 <= Inputs then
    --Both input nodes
    return
  end
  if neuron2 <= Inputs then
    -- Swap output and input
    local temp = neuron1
    neuron1 = neuron2
    neuron2 = temp
  end

  newLink.into = neuron1
  newLink.out = neuron2
  if forceBias then
    newLink.into = Inputs
  end
  
  if containsLink(genome.genes, newLink) then
    return
  end
  newLink.innovation = newInnovation()
  newLink.weight = math.random()*4-2
  
  table.insert(genome.genes, newLink)
end

function nodeMutate(genome)
  if #genome.genes == 0 then
    return
  end

  genome.maxneuron = genome.maxneuron + 1

  local gene = genome.genes[math.random(1,#genome.genes)]
  if not gene.enabled then
    return
  end
  gene.enabled = false
  
  local gene1 = copyGene(gene)
  gene1.out = genome.maxneuron
  gene1.weight = 1.0
  gene1.innovation = newInnovation()
  gene1.enabled = true
  table.insert(genome.genes, gene1)
  
  local gene2 = copyGene(gene)
  gene2.into = genome.maxneuron
  gene2.innovation = newInnovation()
  gene2.enabled = true
  table.insert(genome.genes, gene2)
end

function enableDisableMutate(genome, enable)
  local candidates = {}
  for _,gene in pairs(genome.genes) do
    if gene.enabled == not enable then
      table.insert(candidates, gene)
    end
  end
  
  if #candidates == 0 then
    return
  end
  
  local gene = candidates[math.random(1,#candidates)]
  gene.enabled = not gene.enabled
end

function mutate(genome)
  for mutation,rate in pairs(genome.mutationRates) do
    if math.random(1,2) == 1 then
      genome.mutationRates[mutation] = 0.95*rate
    else
      genome.mutationRates[mutation] = 1.05263*rate
    end
  end

  if math.random() < genome.mutationRates["connections"] then
    pointMutate(genome)
  end
  
  local p = genome.mutationRates["link"]
  while p > 0 do
    if math.random() < p then
      linkMutate(genome, false)
    end
    p = p - 1
  end

  p = genome.mutationRates["bias"]
  while p > 0 do
    if math.random() < p then
      linkMutate(genome, true)
    end
    p = p - 1
  end
  
  p = genome.mutationRates["node"]
  while p > 0 do
    if math.random() < p then
      nodeMutate(genome)
    end
    p = p - 1
  end
  
  p = genome.mutationRates["enable"]
  while p > 0 do
    if math.random() < p then
      enableDisableMutate(genome, true)
    end
    p = p - 1
  end

  p = genome.mutationRates["disable"]
  while p > 0 do
    if math.random() < p then
      enableDisableMutate(genome, false)
    end
    p = p - 1
  end
end

function disjoint(genes1, genes2)
  local i1 = {}
  for i = 1,#genes1 do
    local gene = genes1[i]
    i1[gene.innovation] = true
  end

  local i2 = {}
  for i = 1,#genes2 do
    local gene = genes2[i]
    i2[gene.innovation] = true
  end
  
  local disjointGenes = 0
  for i = 1,#genes1 do
    local gene = genes1[i]
    if not i2[gene.innovation] then
      disjointGenes = disjointGenes+1
    end
  end
  
  for i = 1,#genes2 do
    local gene = genes2[i]
    if not i1[gene.innovation] then
      disjointGenes = disjointGenes+1
    end
  end
  
  local n = math.max(#genes1, #genes2)
  
  return disjointGenes / n
end

function weights(genes1, genes2)
  local i2 = {}
  for i = 1,#genes2 do
    local gene = genes2[i]
    i2[gene.innovation] = gene
  end

  local sum = 0
  local coincident = 0
  for i = 1,#genes1 do
    local gene = genes1[i]
    if i2[gene.innovation] ~= nil then
      local gene2 = i2[gene.innovation]
      sum = sum + math.abs(gene.weight - gene2.weight)
      coincident = coincident + 1
    end
  end
  
  return sum / coincident
end
  
function sameSpecies(genome1, genome2)
  local dd = DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
  local dw = DeltaWeights*weights(genome1.genes, genome2.genes) 
  return dd + dw < DeltaThreshold
end

function rankGlobally()
  local global = {}
  for s = 1,#pool.species do
    local species = pool.species[s]
    for g = 1,#species.genomes do
      table.insert(global, species.genomes[g])
    end
  end
  table.sort(global, function (a,b)
    return (a.fitness < b.fitness)
  end)
  
  for g=1,#global do
    global[g].globalRank = g
  end
end

function calculateAverageFitness(species)
  local total = 0
  
  for g=1,#species.genomes do
    local genome = species.genomes[g]
    total = total + genome.globalRank
  end
  
  species.averageFitness = total / #species.genomes
end

function totalAverageFitness()
  local total = 0
  for s = 1,#pool.species do
    local species = pool.species[s]
    total = total + species.averageFitness
  end

  return total
end

function cullSpecies(cutToOne)
  for s = 1,#pool.species do
    local species = pool.species[s]
    
    table.sort(species.genomes, function (a,b)
      return (a.fitness > b.fitness)
    end)
    
    local remaining = math.ceil(#species.genomes/2)
    if cutToOne then
      remaining = 1
    end
    while #species.genomes > remaining do
      table.remove(species.genomes)
    end
  end
end

function breedChild(species)
  local child = {}
  if math.random() < CrossoverChance then
    g1 = species.genomes[math.random(1, #species.genomes)]
    g2 = species.genomes[math.random(1, #species.genomes)]
    child = crossover(g1, g2)
  else
    g = species.genomes[math.random(1, #species.genomes)]
    child = copyGenome(g)
  end
  
  mutate(child)
  
  return child
end

function removeStaleSpecies()
  local survived = {}

  for s = 1,#pool.species do
    local species = pool.species[s]
    
    table.sort(species.genomes, function (a,b)
      return (a.fitness > b.fitness)
    end)
    
    if species.genomes[1].fitness > species.topFitness then
      species.topFitness = species.genomes[1].fitness
      species.staleness = 0
    else
      species.staleness = species.staleness + 1
    end
    if species.staleness < StaleSpecies or species.topFitness >= pool.maxFitness then
      table.insert(survived, species)
    end
  end

  pool.species = survived
end

function removeWeakSpecies()
  local survived = {}

  local sum = totalAverageFitness()
  for s = 1,#pool.species do
    local species = pool.species[s]
    breed = math.floor(species.averageFitness / sum * Population)
    if breed >= 1 then
      table.insert(survived, species)
    end
  end

  pool.species = survived
end


function addToSpecies(child)
  local foundSpecies = false
  for s=1,#pool.species do
    local species = pool.species[s]
    if not foundSpecies and sameSpecies(child, species.genomes[1]) then
      table.insert(species.genomes, child)
      foundSpecies = true
    end
  end
  
  if not foundSpecies then
    local childSpecies = newSpecies()
    table.insert(childSpecies.genomes, child)
    table.insert(pool.species, childSpecies)
  end
end

function newGeneration()
  cullSpecies(false) -- Cull the bottom half of each species
  rankGlobally()
  removeStaleSpecies()
  rankGlobally()
  for s = 1,#pool.species do
    local species = pool.species[s]
    calculateAverageFitness(species)
  end
  removeWeakSpecies()
  local sum = totalAverageFitness()
  local children = {}
  for s = 1,#pool.species do
    local species = pool.species[s]
    breed = math.floor(species.averageFitness / sum * Population) - 1
    for i=1,breed do
      table.insert(children, breedChild(species))
    end
  end
  cullSpecies(true) -- Cull all but the top member of each species
  while #children + #pool.species < Population do
    local species = pool.species[math.random(1, #pool.species)]
    table.insert(children, breedChild(species))
  end
  for c=1,#children do
    local child = children[c]
    addToSpecies(child)
  end
  
  pool.generation = pool.generation + 1
  
  savePool()
end
  
function initializePool()
  emu.print("Initializing a new pool")

  pool = newPool()

  for i=1,Population do
    basic = basicGenome()
    addToSpecies(basic)
  end
end

function clearJoypad()
  local controller = {}
  for b = 1,#BUTTONS do
    controller[BUTTONS[b]] = false
  end
  joypad.set(1, controller)
end

function nextGenome()
  pool.currentGenome = pool.currentGenome + 1
  if pool.currentGenome > #pool.species[pool.currentSpecies].genomes then
    pool.currentGenome = 1
    pool.currentSpecies = pool.currentSpecies+1
    if pool.currentSpecies > #pool.species then
      newGeneration()
      pool.currentSpecies = 1
    end
  end
end

function fitnessAlreadyMeasured()
  local species = pool.species[pool.currentSpecies]
  local genome = species.genomes[pool.currentGenome]
  
  return genome.fitness ~= 0
end

function writeFile(filename)
  emu.print("Saving pool to " .. filename)

  local file = io.open(WORK_DIR .. "/" .. filename, "w")
  file:write(pool.generation .. "\n")
  file:write(pool.maxFitness .. "\n")
  file:write(#pool.species .. "\n")
  for n,species in pairs(pool.species) do
    file:write(species.topFitness .. "\n")
    file:write(species.staleness .. "\n")
    file:write(#species.genomes .. "\n")
    for m,genome in pairs(species.genomes) do
      file:write(genome.fitness .. "\n")
      file:write(genome.maxneuron .. "\n")
      for mutation,rate in pairs(genome.mutationRates) do
        file:write(mutation .. "\n")
        file:write(rate .. "\n")
      end
      file:write("done\n")

      file:write(#genome.genes .. "\n")
      for l,gene in pairs(genome.genes) do
        file:write(gene.into .. " ")
        file:write(gene.out .. " ")
        file:write(gene.weight .. " ")
        file:write(gene.innovation .. " ")
        if(gene.enabled) then
          file:write("1\n")
        else
          file:write("0\n")
        end
      end
    end
  end
  file:close()
end

function savePool()
  writeFile("backup." .. pool.generation .. "." .. FILENAME)
  writeFile(FILENAME)
end

function loadFile(filename)
  local file = io.open(WORK_DIR .. "/" .. filename, "r")
  if file == nil then
    return false
  end

  emu.print("Loading pool from " .. filename)

  pool = newPool()
  pool.generation = file:read("*number")
  pool.maxFitness = file:read("*number")
  -- forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness))
  local numSpecies = file:read("*number")
  for s=1,numSpecies do
    local species = newSpecies()
    table.insert(pool.species, species)
    species.topFitness = file:read("*number")
    species.staleness = file:read("*number")
    local numGenomes = file:read("*number")
    for g=1,numGenomes do
      local genome = newGenome()
      table.insert(species.genomes, genome)
      genome.fitness = file:read("*number")
      genome.maxneuron = file:read("*number")
      local line = file:read("*line")
      while line ~= "done" do
        genome.mutationRates[line] = file:read("*number")
        line = file:read("*line")
      end
      local numGenes = file:read("*number")
      for n=1,numGenes do
        local gene = newGene()
        table.insert(genome.genes, gene)
        local enabled
        gene.into, gene.out, gene.weight, gene.innovation, enabled = file:read("*number", "*number", "*number", "*number", "*number")
        if enabled == 0 then
          gene.enabled = false
        else
          gene.enabled = true
        end

      end
    end
  end
  file:close()

  while fitnessAlreadyMeasured() do
    nextGenome()
  end

  return true
end
 
function loadPool()
  return loadFile(FILENAME)
end

function playTop()
  local maxfitness = 0
  local maxs, maxg
  for s,species in pairs(pool.species) do
    for g,genome in pairs(species.genomes) do
      if genome.fitness > maxfitness then
        maxfitness = genome.fitness
        maxs = s
        maxg = g
      end
    end
  end
  
  pool.currentSpecies = maxs
  pool.currentGenome = maxg
  pool.maxFitness = maxfitness
end

---- Game ----

-- Enemies.
function GetEnemies()
  local ret = {}
  -- Incoming enemies.
  for addr=0x203,0x253,0x10 do
    local x = memory.readbyte(addr)
    local y = memory.readbyte(addr + 1)
    if x > 0 and y > 0 then
      x = x + DX
      y = y + DY / 2
      ret[#ret+1] = {x=x, y=y}
    end
  end
  -- Enemies standing still.
  local dx = memory.readbyte(0xE5)
  for i=0,9 do
    local x = (dx + 48) % 256 + 16 * i + DX
    local y = 102 + DY / 2
    local mask = memory.readbyte(0xC3 + i)
    while mask > 0 do
      if mask % 2 ~= 0 then
        ret[#ret+1] = {x=x, y=y}
        mask = mask - 1
      end
      mask = mask / 2
      y = y - DY
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
function GetSight(galaxian_gx)
  local sight = {
    x0 = galaxian_gx - SIGHT_RADIUS * DX,
    y0 = Y2 - SIGHT_RADIUS * DY,
  }
  sight.x1 = math.max(galaxian_gx - SIGHT_RADIUS * DX, X1)
  sight.x2 = math.min(galaxian_gx + SIGHT_RADIUS * DX, X2)
  sight.y1 = sight.y0
  sight.y2 = Y2
  return sight
end

-- Tile map.
function GetTileMap(enemies, bullets, sight)
  map = {}

  function Add(gx, gy, val)
    if val ~= 0 and gy < sight.y2 then
      gx = math.max(sight.x1, math.min(sight.x2 - DX, gx))
      gy = math.max(sight.y1, gy)
      if map[gx] == nil then
        map[gx] = {}
      end
      if map[gx][gy] == nil then
        map[gx][gy] = 0
      end
      map[gx][gy] = map[gx][gy] + val
    end
  end

  for _, e in pairs(enemies) do
    local gx = (e.x - X1) - (e.x - X1) % DX + X1
    local gy = (e.y - Y1) - (e.y - Y1) % DY + Y1
    local cx = gx + DX / 2
    local cy = gy + DY / 2
    local pxl = math.max(cx - e.x, 0) / DX
    local pxr = math.max(e.x - cx, 0) / DX
    local pxc = 1 - pxl - pxr
    Add(gx, gy, pxc)
    Add(gx - DX, gy, pxl)
    Add(gx + DX, gy, pxr)
  end
  for _, b in pairs(bullets) do
    local gx = (b.x - X1) - (b.x - X1) % DX + X1
    local gy = (b.y - Y1) - (b.y - Y1) % DY + Y1
    Add(gx, gy, 1)
  end
  return map
end

-- Our missile. nil if not fired.
function GetMissile()
  local x = memory.readbyte(0x283)
  local y = memory.readbyte(0x280)
  if x > 0 and y > 0 then
    return {x=x, y=y}
  else
    return nil
  end
end

function GetScore()
  local score = 0
  for addr=0x6A0,0x6A5 do
    score = score * 10 + AND(memory.readbyte(addr), 0xF)
  end
  return score
end

---- UI ----

SHOW_GRID = false
SHOW_COOR = false
SHOW_TILE_MAP = false
SHOW_OBJECTS = false
SHOW_BANNER = true

function Show(g)
  if SHOW_GRID then
    color = {0xFF, 0xFF, 0xFF, 0x80}
    for x = X1,X2,DX do
      gui.drawline(x, Y1, x, Y2, color)
      if SHOW_COOR then
        local y = 165 + (x % (3 * DX) / DX) * DY
        gui.drawtext(x, y, x)
      end
    end
    for y = Y1,Y2,DY do
      gui.drawline(X1, y, X2, y, color)
      if SHOW_COOR then
        gui.drawtext(5, y, y)
      end
    end
  end

  if SHOW_TILE_MAP then
    for x, row in pairs(g.tile_map) do
      for y, val in pairs(row) do
        gui.drawbox(x, y, x + DX, y + DY, {0xFF, 0, 0, 0x40 * val}, 'clear')
      end
    end
    gui.drawbox(g.galaxian_gx, Y2 - DY, g.galaxian_gx + DX, Y2, {0, 0xFF, 0, 0x80}, 'clear')
  end

  if SHOW_OBJECTS then
    for _, e in pairs(g.enemies) do
      gui.drawbox(e.x - DX, e.y - DY / 2, e.x + DX, e.y + DY / 2, {0xFF, 0, 0, 0x80}, 'clear')
    end
    for _, b in pairs(g.bullets) do
      gui.drawbox(b.x - 4, b.y - 4, b.x + 4, b.y + 4, {0xFF, 0xFF, 0, 0x80}, 'clear')
    end
    galaxian_y = 200
    gui.drawbox(g.galaxian_x - 4, galaxian_y, g.galaxian_x + 4, galaxian_y + 8, 'green')
  end

  if SHOW_BANNER then
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
        pool.currentSpecies .. " genome " .. pool.currentGenome ..
        " (" .. math.floor(measured/total*100) .. "%)", 'black', 'clear')
    gui.drawtext(5, 18, "Fitness: " .. g.score, 'black', 'clear')
    gui.drawtext(100, 18, "Max Fitness: " .. pool.maxFitness, 'black', 'clear')
  end
end

---- Misc ----

function SkipFrames(frames)
  for i=1,frames do
    emu.frameadvance();
  end
end

---- Script starts here ----

emu.print("Running NEAT Galaxian")

-- Reset then enter menu then save.
emu.softreset()
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
INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE);

if not loadPool() then
  initializePool()
end

local controller = nil

while true do
  local g = {}
  g.galaxian_x = (memory.readbyte(0xE4) + 124) % 256 + 4
  g.galaxian_gx = (g.galaxian_x - X1) - (g.galaxian_x - X1) % DX + X1
  g.enemies = GetEnemies()
  g.bullets = GetBullets()
  g.missile = GetMissile()
  g.score = GetScore()
  g.lifes = memory.readbyte(0x42)
  g.sight = GetSight(g.galaxian_gx)
  g.tile_map = GetTileMap(g.enemies, g.bullets, g.sight)

  local species = pool.species[pool.currentSpecies]
  local genome = species.genomes[pool.currentGenome]
  if genome.network == nil then
    generateNetwork(genome)
  end
  if controller == nil or pool.currentFrame % 5 == 0 then
    controller = evaluateNetwork(genome.network, getInputs(g))
  end
  joypad.set(1, controller)

  -- Reset if dead.
  if g.lifes < 2 then
    genome.fitness = g.score + pool.currentFrame + 1
    
    if genome.fitness > pool.maxFitness then
      pool.maxFitness = genome.fitness
      savePool()
    end
    
    emu.print("Gen " .. pool.generation .. " species " .. pool.currentSpecies ..
        " genome " .. pool.currentGenome .. " fitness: " .. genome.fitness)

    savestate.load(INIT_STATE);
    pool.currentFrame = 0
    clearJoypad()

    pool.currentSpecies = 1
    pool.currentGenome = 1
    while fitnessAlreadyMeasured() do
      nextGenome()
    end
    
    local species = pool.species[pool.currentSpecies]
    local genome = species.genomes[pool.currentGenome]
    generateNetwork(genome)
  end

  Show(g)

  pool.currentFrame = pool.currentFrame + 1
  emu.frameadvance();
end
