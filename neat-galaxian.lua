-- Galaxian using Neural Network and Genetic Algorithm
-- Author: kelvinlau
--
-- Inspired by MarI/O by SethBling.
-- Intended for use on the FCEUX emulator.
-- Enter a level, then load this script.
--
-- TODO ideas:
-- * Run on a cluster using GUI-less emulation.
-- * Time traveling.
-- * Learn from human (back propagation).
-- * galaxian_x + dx instead of galaxian_gx + dx

---- Configs ----

WORK_DIR = "Z:/Users/kelvinlau/neat"
FILENAME = "galaxian.v4.pool"
PLAY_TOP = false

---- Game constants ----

BUTTONS = {"A", "left", "right"}

X1 = 16
X2 = 240
DX = 8
Y1 = 42
Y2 = 222
DY = 12

SIGHT_Y = 8
SIGHT_X = 8

---- NN and GA constants ----

NUM_SNAPSHOTS = 3
NUM_TILE_MAP_NODES = (SIGHT_X*2) * SIGHT_Y
NUM_SPECIAL_INPUT_NODES = 4
NUM_INPUT_NODES = NUM_TILE_MAP_NODES * NUM_SNAPSHOTS + NUM_SPECIAL_INPUT_NODES
NUM_OUTPUT_NODES = #BUTTONS

POPULATION = 300
DELTA_DISJOINT = 2.0
DELTA_WEIGHTS = 0.4
DELTA_THRESHOLD = 1.0

STALE_SPECIES = 15

MUTATE_CONNECTION_CHANCE = 0.25
PERTURB_CHANCE = 0.90
CROSSOVER_CHANCE = 0.75
LINK_MUTATION_CHANCE = 2.0
NODE_MUTATION_CHANCE = 0.60
SPECIAL_MUTATION_CHANCE = 0.40
STEP_SIZE = 0.1
DISABLE_MUTATION_CHANCE = 0.4
ENABLE_MUTATION_CHANCE = 0.2

MAX_NODES = 1000000

----

function GetInputs(g)
  local inputs = {}
  -- tile_map (last several snapshots)
  -- TODO: enemy type?
  for id = 1,NUM_TILE_MAP_NODES * NUM_SNAPSHOTS do
    inputs[id] = 0
  end
  for map_id, tile_map in pairs(g.prev_tile_maps) do
    for x, row in pairs(tile_map) do
      for y, val in pairs(row) do
        if (map_id == 0 or
            (g.sight.x1 <= x and x < g.sight.x2 and
             g.sight.y1 <= y and y < g.sight.y2)) then
          local ix = (x - g.sight.x0) / DX
          local iy = (y - g.sight.y0) / DY
          local id = map_id * NUM_TILE_MAP_NODES + iy * (2 * SIGHT_X) + ix + 1
          if 1 <= id and id <= #inputs then
            inputs[id] = val
          else
            emu.print("Invalid tile:", map_id, x, y, val, ix, iy, id, g.sight)
          end
        end
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
  -- bias input neuron
  inputs[#inputs+1] = 1
  return inputs
end

function Sigmoid(x)
  return 2/(1+math.exp(-4.9*x))-1
end

function NewInnovation()
  pool.innovation = pool.innovation + 1
  return pool.innovation
end

function NewPool()
  local pool = {}
  pool.species = {}
  pool.generation = 0
  pool.innovation = NUM_OUTPUT_NODES
  pool.cur_species = 1
  pool.cur_genome = 1
  pool.cur_frame = 0
  pool.max_fitness = 0
  
  return pool
end

function NewSpecies()
  local species = {}
  species.top_fitness = 0
  species.staleness = 0
  species.genomes = {}
  species.avg_fitness = 0
  
  return species
end

function NewGenome()
  local genome = {}
  genome.genes = {}
  genome.fitness = 0
  genome.max_neuron = 0
  genome.global_rank = 0
  genome.mutation_rates = {}
  genome.mutation_rates["connections"] = MUTATE_CONNECTION_CHANCE
  genome.mutation_rates["link"] = LINK_MUTATION_CHANCE
  genome.mutation_rates["special"] = SPECIAL_MUTATION_CHANCE
  genome.mutation_rates["node"] = NODE_MUTATION_CHANCE
  genome.mutation_rates["enable"] = ENABLE_MUTATION_CHANCE
  genome.mutation_rates["disable"] = DISABLE_MUTATION_CHANCE
  genome.mutation_rates["step"] = STEP_SIZE
  
  return genome
end

function CopyGenome(genome)
  local genome2 = NewGenome()
  for g=1,#genome.genes do
    table.insert(genome2.genes, CopyGene(genome.genes[g]))
  end
  genome2.max_neuron = genome.max_neuron
  for mutation,rate in pairs(genome.mutation_rates) do
    genome2.mutation_rates[mutation] = rate
  end
  
  return genome2
end

function BasicGenome()
  local genome = NewGenome()

  genome.max_neuron = NUM_INPUT_NODES
  Mutate(genome)
  
  return genome
end

function NewGene()
  local gene = {}
  gene.into = 0
  gene.out = 0
  gene.weight = 0.0
  gene.enabled = true
  gene.innovation = 0
  
  return gene
end

function CopyGene(gene)
  local gene2 = NewGene()
  gene2.into = gene.into
  gene2.out = gene.out
  gene2.weight = gene.weight
  gene2.enabled = gene.enabled
  gene2.innovation = gene.innovation
  
  return gene2
end

function NewNeuron()
  local neuron = {}
  neuron.incoming = {}
  neuron.value = 0.0
  
  return neuron
end

function GenerateNetwork(genome)
  local network = {}
  network.neurons = {}
  
  for i=1,NUM_INPUT_NODES do
    network.neurons[i] = NewNeuron()
  end
  
  for o=1,NUM_OUTPUT_NODES do
    network.neurons[MAX_NODES+o] = NewNeuron()
  end
  
  table.sort(genome.genes, function (a,b)
    return (a.out < b.out)
  end)
  for i=1,#genome.genes do
    local gene = genome.genes[i]
    if gene.enabled then
      if network.neurons[gene.out] == nil then
        network.neurons[gene.out] = NewNeuron()
      end
      local neuron = network.neurons[gene.out]
      table.insert(neuron.incoming, gene)
      if network.neurons[gene.into] == nil then
        network.neurons[gene.into] = NewNeuron()
      end
    end
  end
  
  genome.network = network
end

function EvaluateNetwork(network, inputs)
  if #inputs ~= NUM_INPUT_NODES then
    emu.print("Incorrect number of neural network inputs.", #inputs, NUM_INPUT_NODES)
    return {}
  end
  
  for i=1,NUM_INPUT_NODES do
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
      neuron.value = Sigmoid(sum)
    end
  end
  
  local outputs = {}
  for o=1,NUM_OUTPUT_NODES do
    local button = BUTTONS[o]
    if network.neurons[MAX_NODES+o].value > 0 then
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

function Crossover(g1, g2)
  -- Make sure g1 is the higher fitness genome
  if g2.fitness > g1.fitness then
    local gx = g1
    g1 = g2
    g2 = gx
  end

  local child = NewGenome()
  
  local innovations2 = {}
  for i=1,#g2.genes do
    local gene = g2.genes[i]
    innovations2[gene.innovation] = gene
  end
  
  for i=1,#g1.genes do
    local gene1 = g1.genes[i]
    local gene2 = innovations2[gene1.innovation]
    if gene2 ~= nil and math.random(2) == 1 and gene2.enabled then
      table.insert(child.genes, CopyGene(gene2))
    else
      table.insert(child.genes, CopyGene(gene1))
    end
  end
  
  child.max_neuron = math.max(g1.max_neuron, g2.max_neuron)
  
  for mutation,rate in pairs(g1.mutation_rates) do
    child.mutation_rates[mutation] = rate
  end
  
  return child
end

function RandomNeuron(genes, include_input)
  local neurons = {}
  if include_input then
    for i=1,NUM_INPUT_NODES do
      neurons[i] = true
    end
  end
  for o=1,NUM_OUTPUT_NODES do
    neurons[MAX_NODES+o] = true
  end
  for i=1,#genes do
    if include_input or genes[i].into > NUM_INPUT_NODES then
      neurons[genes[i].into] = true
    end
    if include_input or genes[i].out > NUM_INPUT_NODES then
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

function ContainsLink(genes, link)
  for i=1,#genes do
    local gene = genes[i]
    if gene.into == link.into and gene.out == link.out then
      return true
    end
  end
end

function PointMutate(genome)
  local step = genome.mutation_rates["step"]
  
  for i=1,#genome.genes do
    local gene = genome.genes[i]
    if math.random() < PERTURB_CHANCE then
      gene.weight = gene.weight + math.random() * step*2 - step
    else
      gene.weight = math.random()*4-2
    end
  end
end

function LinkMutate(genome, force_special_input_nodes)
  local neuron1 = RandomNeuron(genome.genes, true)
  local neuron2 = RandomNeuron(genome.genes, false)
   
  local new_link = NewGene()
  if neuron1 <= NUM_INPUT_NODES and neuron2 <= NUM_INPUT_NODES then
    -- Both input nodes, it's impossible
    return
  end
  if neuron2 <= NUM_INPUT_NODES then
    -- Swap output and input
    local temp = neuron1
    neuron1 = neuron2
    neuron2 = temp
  end

  new_link.into = neuron1
  new_link.out = neuron2
  if force_special_input_nodes then
    new_link.into = NUM_INPUT_NODES - math.random(0, NUM_SPECIAL_INPUT_NODES)
  end
  
  if ContainsLink(genome.genes, new_link) then
    return
  end
  new_link.innovation = NewInnovation()
  new_link.weight = math.random()*4-2
  
  table.insert(genome.genes, new_link)
end

function NodeMutate(genome)
  if #genome.genes == 0 then
    return
  end

  genome.max_neuron = genome.max_neuron + 1

  local gene = genome.genes[math.random(1,#genome.genes)]
  if not gene.enabled then
    return
  end
  gene.enabled = false
  
  local gene1 = CopyGene(gene)
  gene1.out = genome.max_neuron
  gene1.weight = 1.0
  gene1.innovation = NewInnovation()
  gene1.enabled = true
  table.insert(genome.genes, gene1)
  
  local gene2 = CopyGene(gene)
  gene2.into = genome.max_neuron
  gene2.innovation = NewInnovation()
  gene2.enabled = true
  table.insert(genome.genes, gene2)
end

function EnableDisableMutate(genome, enable)
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

function Mutate(genome)
  for mutation,rate in pairs(genome.mutation_rates) do
    if math.random(1,2) == 1 then
      genome.mutation_rates[mutation] = 0.95*rate
    else
      genome.mutation_rates[mutation] = 1.05263*rate
    end
  end

  if math.random() < genome.mutation_rates["connections"] then
    PointMutate(genome)
  end
  
  local p = genome.mutation_rates["link"]
  while p > 0 do
    if math.random() < p then
      LinkMutate(genome, false)
    end
    p = p - 1
  end

  p = genome.mutation_rates["special"]
  while p > 0 do
    if math.random() < p then
      LinkMutate(genome, true)
    end
    p = p - 1
  end
  
  p = genome.mutation_rates["node"]
  while p > 0 do
    if math.random() < p then
      NodeMutate(genome)
    end
    p = p - 1
  end
  
  p = genome.mutation_rates["enable"]
  while p > 0 do
    if math.random() < p then
      EnableDisableMutate(genome, true)
    end
    p = p - 1
  end

  p = genome.mutation_rates["disable"]
  while p > 0 do
    if math.random() < p then
      EnableDisableMutate(genome, false)
    end
    p = p - 1
  end
end

function Disjoint(genes1, genes2)
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
  
  local num_disjoint_genes = 0
  for i = 1,#genes1 do
    local gene = genes1[i]
    if not i2[gene.innovation] then
      num_disjoint_genes = num_disjoint_genes+1
    end
  end
  
  for i = 1,#genes2 do
    local gene = genes2[i]
    if not i1[gene.innovation] then
      num_disjoint_genes = num_disjoint_genes+1
    end
  end
  
  local n = math.max(#genes1, #genes2)
  
  return num_disjoint_genes / n
end

function Weights(genes1, genes2)
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
  
function SameSpecies(genome1, genome2)
  local dd = DELTA_DISJOINT*Disjoint(genome1.genes, genome2.genes)
  local dw = DELTA_WEIGHTS*Weights(genome1.genes, genome2.genes) 
  return dd + dw < DELTA_THRESHOLD
end

function RankGlobally()
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
    global[g].global_rank = g
  end
end

function CalculateAverageFitness(species)
  local total = 0
  
  for g=1,#species.genomes do
    local genome = species.genomes[g]
    total = total + genome.global_rank
  end
  
  species.avg_fitness = total / #species.genomes
end

function TotalAverageFitness()
  local total = 0
  for s = 1,#pool.species do
    local species = pool.species[s]
    total = total + species.avg_fitness
  end

  return total
end

function CullSpecies(cut_to_one)
  for s = 1,#pool.species do
    local species = pool.species[s]
    
    table.sort(species.genomes, function (a,b)
      return (a.fitness > b.fitness)
    end)
    
    local remaining = math.ceil(#species.genomes/2)
    if cut_to_one then
      remaining = 1
    end
    while #species.genomes > remaining do
      table.remove(species.genomes)
    end
  end
end

function BreedChild(species)
  local child = {}
  if math.random() < CROSSOVER_CHANCE then
    g1 = species.genomes[math.random(1, #species.genomes)]
    g2 = species.genomes[math.random(1, #species.genomes)]
    child = Crossover(g1, g2)
  else
    g = species.genomes[math.random(1, #species.genomes)]
    child = CopyGenome(g)
  end
  
  Mutate(child)
  
  return child
end

function RemoveStaleSpecies()
  local survived = {}

  for s = 1,#pool.species do
    local species = pool.species[s]
    
    table.sort(species.genomes, function (a,b)
      return (a.fitness > b.fitness)
    end)
    
    if species.genomes[1].fitness > species.top_fitness then
      species.top_fitness = species.genomes[1].fitness
      species.staleness = 0
    else
      species.staleness = species.staleness + 1
    end
    if species.staleness < STALE_SPECIES or species.top_fitness >= pool.max_fitness then
      table.insert(survived, species)
    end
  end

  pool.species = survived
end

function RemoveWeakSpecies()
  local survived = {}

  local sum = TotalAverageFitness()
  for s = 1,#pool.species do
    local species = pool.species[s]
    breed = math.floor(species.avg_fitness / sum * POPULATION)
    if breed >= 1 then
      table.insert(survived, species)
    end
  end

  pool.species = survived
end


function AddToSpecies(child)
  local found_species = false
  for s=1,#pool.species do
    local species = pool.species[s]
    if not found_species and SameSpecies(child, species.genomes[1]) then
      table.insert(species.genomes, child)
      found_species = true
    end
  end
  
  if not found_species then
    local child_species = NewSpecies()
    table.insert(child_species.genomes, child)
    table.insert(pool.species, child_species)
  end
end

function NewGeneration()
  CullSpecies(false) -- Cull the bottom half of each species
  RankGlobally()
  RemoveStaleSpecies()
  RankGlobally()
  for s = 1,#pool.species do
    local species = pool.species[s]
    CalculateAverageFitness(species)
  end
  RemoveWeakSpecies()
  local sum = TotalAverageFitness()
  local children = {}
  for s = 1,#pool.species do
    local species = pool.species[s]
    breed = math.floor(species.avg_fitness / sum * POPULATION) - 1
    for i=1,breed do
      table.insert(children, BreedChild(species))
    end
  end
  CullSpecies(true) -- Cull all but the top member of each species
  while #children + #pool.species < POPULATION do
    local species = pool.species[math.random(1, #pool.species)]
    table.insert(children, BreedChild(species))
  end
  for c=1,#children do
    local child = children[c]
    AddToSpecies(child)
  end
  
  pool.generation = pool.generation + 1
  
  SavePool()
end
  
function InitializePool()
  emu.print("Initializing a new pool")

  pool = NewPool()

  for i=1,POPULATION do
    basic = BasicGenome()
    AddToSpecies(basic)
  end
end

function ClearJoypad()
  local controller = {}
  for b = 1,#BUTTONS do
    controller[BUTTONS[b]] = false
  end
  joypad.set(1, controller)
end

function NextGenome()
  pool.cur_genome = pool.cur_genome + 1
  if pool.cur_genome > #pool.species[pool.cur_species].genomes then
    pool.cur_genome = 1
    pool.cur_species = pool.cur_species+1
    if pool.cur_species > #pool.species then
      NewGeneration()
      pool.cur_species = 1
    end
  end
end

function FitnessAlreadyMeasured()
  local species = pool.species[pool.cur_species]
  local genome = species.genomes[pool.cur_genome]
  return genome.fitness ~= 0
end

function WriteFile(filename)
  emu.print("Saving pool to " .. filename)

  local file = io.open(WORK_DIR .. "/" .. filename, "w")
  file:write(pool.generation .. "\n")
  file:write(pool.max_fitness .. "\n")
  file:write(#pool.species .. "\n")
  for n,species in pairs(pool.species) do
    file:write(species.top_fitness .. "\n")
    file:write(species.staleness .. "\n")
    file:write(#species.genomes .. "\n")
    for m,genome in pairs(species.genomes) do
      file:write(genome.fitness .. "\n")
      file:write(genome.max_neuron .. "\n")
      for mutation,rate in pairs(genome.mutation_rates) do
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

function SavePool()
  WriteFile("backup." .. pool.generation .. "." .. FILENAME)
  WriteFile(FILENAME)
end

function LoadFile(filename)
  local file = io.open(WORK_DIR .. "/" .. filename, "r")
  if file == nil then
    return false
  end

  emu.print("Loading pool from " .. filename)

  pool = NewPool()
  pool.generation = file:read("*number")
  pool.max_fitness = file:read("*number")
  local num_species = file:read("*number")
  for s=1,num_species do
    local species = NewSpecies()
    table.insert(pool.species, species)
    species.top_fitness = file:read("*number")
    species.staleness = file:read("*number")
    local num_genomes = file:read("*number")
    for g=1,num_genomes do
      local genome = NewGenome()
      table.insert(species.genomes, genome)
      genome.fitness = file:read("*number")
      genome.max_neuron = file:read("*number")
      local line = file:read("*line")
      while line ~= "done" do
        genome.mutation_rates[line] = file:read("*number")
        line = file:read("*line")
      end
      local num_genes = file:read("*number")
      for n=1,num_genes do
        local gene = NewGene()
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

  while FitnessAlreadyMeasured() do
    NextGenome()
  end

  return true
end
 
function LoadPool()
  return LoadFile(FILENAME)
end

function PlayTop()
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
  
  pool.cur_species = maxs
  pool.cur_genome = maxg
  pool.max_fitness = maxfitness
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
      ret[#ret+1] = {x=x, y=y, w=1}
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
        ret[#ret+1] = {x=x, y=y, w=0.2}
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
    x0 = galaxian_gx - SIGHT_X * DX,
    y0 = Y2 - SIGHT_Y * DY,
  }
  sight.x1 = math.max(galaxian_gx - SIGHT_X * DX, X1)
  sight.x2 = math.min(galaxian_gx + SIGHT_X * DX, X2)
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
    Add(gx,      gy, e.w * pxc)
    Add(gx - DX, gy, e.w * pxl)
    Add(gx + DX, gy, e.w * pxr)
  end
  for _, b in pairs(bullets) do
    local gx = (b.x - X1) - (b.x - X1) % DX + X1
    local gy = (b.y - Y1) - (b.y - Y1) % DY + Y1
    Add(gx, gy, -1)  -- TODO: better add a layer for bullets than using -1 here.
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
SHOW_TILE_MAP = true
SHOW_OBJECTS = false
SHOW_BANNER = true
SHOW_AI_VISION = false

function Show(g, genome)
  if SHOW_AI_VISION then
    gui.drawbox(0, 0, 256, 256, 'black', 'clear')
  end
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
        local color = nil
        if val >= 0 then
          color = {0, 0, 0xFF, 0x40 * val} -- enemies are blue
        else
          color = {0xFF, 0, 0, 0x40 * -val} -- bullets are red
        end
        gui.drawbox(x, y, x + DX, y + DY, color, 'clear')
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
        pool.cur_species .. " genome " .. pool.cur_genome ..
        " (" .. math.floor(measured/total*100) .. "%)", 'black', 'clear')
    gui.drawtext(5, 18, "Fitness: " .. genome.fitness, 'black', 'clear')
    gui.drawtext(100, 18, "Max Fitness: " .. pool.max_fitness, 'black', 'clear')
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
INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE);

if not LoadPool() then
  InitializePool()
end

if PLAY_TOP then
  -- PlayTop()
  pool.cur_species = 11
  pool.cur_genome = 1
  emu.speedmode("normal")
end

local controller = nil
local prev_tile_maps = {}

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
  g.prev_tile_maps = prev_tile_maps
  g.prev_tile_maps[0] = tile_map

  local species = pool.species[pool.cur_species]
  local genome = species.genomes[pool.cur_genome]

  genome.fitness = g.score + math.floor(pool.cur_frame / 20) + 1
  if genome.network == nil then
    GenerateNetwork(genome)
  end

  -- React every 10 frames.
  if controller == nil or pool.cur_frame % 10 == 0 then
    controller = EvaluateNetwork(genome.network, GetInputs(g))
  end
  joypad.set(1, controller)

  -- Add a snapshot for every 30 frames.
  if pool.cur_frame % 30 == 0 then
    for i = NUM_SNAPSHOTS-1,2 do
      prev_tile_maps[i] = prev_tile_maps[i-1]
    end
    prev_tile_maps[1] = g.tile_map
  end

  -- Reset if dead.
  if g.lifes < 2 then
    if genome.fitness > pool.max_fitness then
      pool.max_fitness = genome.fitness
      SavePool()
    end
    
    emu.print("Gen " .. pool.generation .. " species " .. pool.cur_species ..
        " genome " .. pool.cur_genome .. " neurons: " .. genome.max_neuron ..
        " genes: " .. #genome.genes ..  " fitness: " .. genome.fitness ..
        " score: " .. g.score)

    savestate.load(INIT_STATE);
    pool.cur_frame = 0
    ClearJoypad()

    pool.cur_species = 1
    pool.cur_genome = 1
    while FitnessAlreadyMeasured() do
      NextGenome()
    end
    
    local species = pool.species[pool.cur_species]
    local genome = species.genomes[pool.cur_genome]
    GenerateNetwork(genome)
  end

  Show(g, genome)

  pool.cur_frame = pool.cur_frame + 1
  emu.frameadvance();
end
