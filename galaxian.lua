-- Galaxian using Neural Network and Genetic Algorithm
-- Author: kelvinlau
--
-- Inspired by MarI/O by SethBling.
-- Intended for use on the FCEUX emulator.
-- Load this script, it will reset the game and start evolving.
--
-- TODO ideas:
-- * Run on a cluster using GUI-less emulation.
-- * Time traveling.
-- * Learn from human (back propagation).
-- * Hidden layers in BasicGenome?
-- * Incoming enemies and bullets' coordinates as inputs (v7).
-- * Recurrent Neural Network.
-- * Use IUP to build forms.

require("game")

---- Configs ----

SHOW_GRID = false
SHOW_COOR = false
SHOW_OBJECTS = false
SHOW_TILE_MAP = true
SHOW_BANNER = true
SHOW_AI_VISION = true
SHOW_NETWORK = true

WORK_DIR = "Z:/Users/kelvinlau/neat"
FILENAME = "galaxian.v8.pool"

PLAY_TOP = false
READ_ONLY = false
HUMAN_PLAY = false
DEBUG = false

---- Neuron Evolution ----

NUM_OUTPUT_NEURONS = 3

POPULATION = 300
DELTA_DISJOINT = 2.0
DELTA_WEIGHTS = 0.4
DELTA_THRESHOLD = 1.0

STALE_SPECIES = 15

MUTATE_CONNECTION_CHANCE = 0.25
PERTURB_CHANCE = 0.90
CROSSOVER_CHANCE = 0.75
GENE_MUTATION_CHANCE = 2.0
NEURON_MUTATION_CHANCE = 0.60
BIAS_MUTATION_CHANCE = 0.40
STEP_SIZE = 0.1
DISABLE_MUTATION_CHANCE = 0.4
ENABLE_MUTATION_CHANCE = 0.2

recent_inputs = {}

function AddToRecentInputs(inputs)
  for id, _ in pairs(inputs) do
    recent_inputs[id] = pool.generation
  end
  recent_inputs["m"] = pool.generation
  recent_inputs["gx"] = pool.generation
  recent_inputs["bias"] = pool.generation
  recent_inputs["rnd"] = pool.generation
end

function Sigmoid(x)
  return 2/(1+math.exp(-4.9*x))-1
end

function ReLu(x)
  return math.max(x, 0)
end

function NewInnovation()
  pool.innovation = pool.innovation + 1
  return pool.innovation
end

function NewPool()
  local pool = {}
  pool.species = {}
  pool.generation = 0
  pool.innovation = NUM_OUTPUT_NEURONS
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
  genome.hidden_neurons = 0
  genome.global_rank = 0
  genome.mutation_rates = {}
  genome.mutation_rates["connections"] = MUTATE_CONNECTION_CHANCE
  genome.mutation_rates["gene"] = GENE_MUTATION_CHANCE
  genome.mutation_rates["bias"] = BIAS_MUTATION_CHANCE
  genome.mutation_rates["neuron"] = NEURON_MUTATION_CHANCE
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
  genome2.hidden_neurons = genome.hidden_neurons
  for mutation,rate in pairs(genome.mutation_rates) do
    genome2.mutation_rates[mutation] = rate
  end
  
  return genome2
end

function BasicGenome()
  local genome = NewGenome()

  Mutate(genome)
  
  return genome
end

function NewGene()
  local gene = {}
  gene.src = nil
  gene.out = nil
  gene.weight = 0.0
  gene.enabled = true
  gene.innovation = 0
  
  return gene
end

function CopyGene(gene)
  local gene2 = NewGene()
  gene2.src = gene.src
  gene2.out = gene.out
  gene2.weight = gene.weight
  gene2.enabled = gene.enabled
  gene2.innovation = gene.innovation
  
  return gene2
end

function NewNeuron()
  local neuron = {}
  neuron.incoming = {}
  neuron.outgoing = {}
  neuron.value = 0.0
  
  return neuron
end

function GenerateNetwork(genome)
  local network = {}
  network.neurons = {}
  
  for o=1,NUM_OUTPUT_NEURONS do
    network.neurons["o"..o] = NewNeuron()
  end
  network.num_neurons = NUM_OUTPUT_NEURONS
  
  table.sort(genome.genes, function (a,b)
    return (a.out < b.out)
  end)
  for i=1,#genome.genes do
    local gene = genome.genes[i]
    if gene.enabled then
      if network.neurons[gene.out] == nil then
        network.neurons[gene.out] = NewNeuron()
        network.num_neurons = network.num_neurons + 1
      end
      local out = network.neurons[gene.out]
      table.insert(out.incoming, gene)
      if network.neurons[gene.src] == nil then
        network.neurons[gene.src] = NewNeuron()
        network.num_neurons = network.num_neurons + 1
      end
      local src = network.neurons[gene.src]
      table.insert(src.outgoing, gene)
    end
  end
  -- emu.print(network)

  -- Topological sort.
  local deg = {}
  local queue = {}
  local head = 1
  for id, neuron in pairs(network.neurons) do
    deg[id] = #neuron.incoming
    if deg[id] == 0 then
      queue[#queue+1] = id
    end
  end
  while head < #queue do
    local neuron = network.neurons[queue[head]]
    head = head + 1
    for _, gene in pairs(neuron.outgoing) do
      local out = gene.out
      deg[out] = deg[out] - 1
      if deg[out] == 0 then
        queue[#queue+1] = out
      end
    end
  end

  network.topological_ids = queue
  if DEBUG then
    for i = 1, #network.topological_ids do
      local id = network.topological_ids[i]
      local incoming = network.neurons[id].incoming
      emu.print(i, id, #incoming, network.neurons[id])
    end
    emu.print("")
  end
  
  genome.network = network
end

function EvaluateNetwork(network, inputs)
  for id, val in pairs(inputs) do
    if network.neurons[id] ~= nil then
      network.neurons[id].value = val
    end
  end

  -- Feed forward.
  -- TODO: Skip input neurons.
  for i = 1, #network.topological_ids do
    local id = network.topological_ids[i]
    local neuron = network.neurons[id]
    if #neuron.incoming > 0 then
      local sum = 0
      for j = 1,#neuron.incoming do
        local gene = neuron.incoming[j]
        local src = network.neurons[gene.src]
        sum = sum + gene.weight * src.value
      end
      neuron.value = ReLu(sum)
    end
  end
  
  local outputs = {}
  local l = network.neurons["o1"].value
  local r = network.neurons["o2"].value
  local d = r-l
  local threshold = 0.1
  if l > r + threshold then
    outputs["left"] = true
    outputs["right"] = false
  elseif r > l + threshold then
    outputs["left"] = false
    outputs["right"] = true
  else
    outputs["left"] = false
    outputs["right"] = false
  end
  local fire = network.neurons["o3"].value
  if fire > 0 then
    outputs["A"] = true
  else
    outputs["A"] = false
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
  
  child.hidden_neurons = math.max(g1.hidden_neurons, g2.hidden_neurons)
  
  for mutation,rate in pairs(g1.mutation_rates) do
    child.mutation_rates[mutation] = rate
  end
  
  return child
end

function IsInputNeuron(id)
  return not IsHiddenNeuron(id) and not IsOutputNeuron(id)
end

function IsHiddenNeuron(id)
  return id:sub(1, 1) == "h"
end

function IsOutputNeuron(id)
  return id:sub(1, 1) == "o"
end

function RandomNeuron(genes, include_input, include_output)
  local neurons = {}
  if include_input then
    for id, last_seen_generation in pairs(recent_inputs) do
      if last_seen_generation >= pool.generation-2 then
        neurons[id] = true
      end
    end
  end
  if include_output then
    for o = 1, NUM_OUTPUT_NEURONS do
      neurons["o"..o] = true
    end
  end
  for i = 1, #genes do
    if IsHiddenNeuron(genes[i].src) then
      neurons[genes[i].src] = true
    end
    if IsHiddenNeuron(genes[i].out) then
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
  
  return nil
end

function ContainsGene(genes, g)
  for i=1,#genes do
    local gene = genes[i]
    if gene.src == g.src and gene.out == g.out then
      return true
    end
  end
  return false
end

function PointMutate(genome)
  local step = genome.mutation_rates["step"]
  
  for i=1,#genome.genes do
    local gene = genome.genes[i]
    if math.random() < PERTURB_CHANCE then
      gene.weight = gene.weight + step * (math.random()*2-1)
    else
      gene.weight = math.random()*4-2
    end
  end
end

function GeneMutate(genome, force_bias)
  local new_gene = NewGene()
  if force_bias then
    new_gene.src = "bias"
  else
    new_gene.src = RandomNeuron(genome.genes, true, false)
  end
  new_gene.out = RandomNeuron(genome.genes, false, true)

  
  if ContainsGene(genome.genes, new_gene) then
    return
  end
  new_gene.innovation = NewInnovation()
  new_gene.weight = math.random()*4-2
  
  table.insert(genome.genes, new_gene)
end

function NeuronMutate(genome)
  if #genome.genes == 0 then
    return
  end

  genome.hidden_neurons = genome.hidden_neurons + 1
  local hid = string.format("h%03d", genome.hidden_neurons)

  local gene = genome.genes[math.random(1,#genome.genes)]
  if not gene.enabled then
    return
  end
  gene.enabled = false
  
  local gene1 = CopyGene(gene)
  gene1.out = hid
  gene1.weight = 1.0
  gene1.innovation = NewInnovation()
  gene1.enabled = true
  table.insert(genome.genes, gene1)
  
  local gene2 = CopyGene(gene)
  gene2.src = hid
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
  
  local p = genome.mutation_rates["gene"]
  while p > 0 do
    if math.random() < p then
      GeneMutate(genome, false)
    end
    p = p - 1
  end

  p = genome.mutation_rates["bias"]
  while p > 0 do
    if math.random() < p then
      GeneMutate(genome, true)
    end
    p = p - 1
  end
  
  p = genome.mutation_rates["neuron"]
  while p > 0 do
    if math.random() < p then
      NeuronMutate(genome)
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

  AddToRecentInputs({})

  for i=1,POPULATION do
    basic = BasicGenome()
    AddToSpecies(basic)
  end
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
      file:write(genome.hidden_neurons .. "\n")
      for mutation,rate in pairs(genome.mutation_rates) do
        file:write(mutation .. "\n")
        file:write(rate .. "\n")
      end
      file:write("done\n")

      file:write(#genome.genes .. "\n")
      for l,gene in pairs(genome.genes) do
        file:write(gene.src .. "\n")
        file:write(gene.out .. "\n")
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
  if READ_ONLY then
    return
  end
  WriteFile(string.format("backup.%04d.%s", pool.generation, FILENAME))
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
      genome.hidden_neurons = file:read("*number")
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
        file:read("*line")
        gene.src = file:read("*line")
        gene.out = file:read("*line")
        gene.weight, gene.innovation, enabled = file:read("*number", "*number", "*number")
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

---- Misc ----

function ClearJoypad()
  local controller = {}
  controller["left"] = false
  controller["right"] = false
  controller["A"] = false
  joypad.set(1, controller)
end

---- Script starts here ----

emu.print("Running NEAT Galaxian")
emu.print("Num input neurons:", #ALL_INPUTS)

-- Reset then enter menu then save.
Reset()
INIT_STATE = savestate.create(9)
savestate.save(INIT_STATE);

if not LoadPool() then
  InitializePool()
end

if PLAY_TOP then
  PlayTop()
  emu.speedmode("normal")
end

if HUMAN_PLAY then
  emu.speedmode("normal")
end

local controller = nil
local recent_games = {}
local survived_incomings = 0

while true do
  local g = GetGame()

  recent_games[0] = g

  local species = pool.species[pool.cur_species]
  local genome = species.genomes[pool.cur_genome]

  genome.fitness = math.max(g.score + survived_incomings * 100, 1)
  if genome.network == nil then
    GenerateNetwork(genome)
  end

  -- React every 10 frames.
  if controller == nil or pool.cur_frame % 10 == 0 then
    local inputs = GetInputs(recent_games)
    AddToRecentInputs(inputs)
    controller = EvaluateNetwork(genome.network, inputs)
  end
  if not HUMAN_PLAY then
    joypad.set(1, controller)
  end

  -- Add a snapshot for every 30 frames.
  if pool.cur_frame % 30 == 0 then
    survived_incomings = survived_incomings + GetSurvivedIncomings(recent_games)
    for i = NUM_SNAPSHOTS-1,2,-1 do
      if recent_games[i-1] ~= nil then
        recent_games[i] = recent_games[i-1]
      end
    end
    recent_games[1] = g
  end

  pool.cur_frame = pool.cur_frame + 1

  Show(recent_games, genome)

  -- Reset if dead.
  if g.lifes < 2 then
    if genome.fitness > pool.max_fitness then
      pool.max_fitness = genome.fitness
      SavePool()
    end
    
    emu.print("Gen " .. pool.generation .. " species " .. pool.cur_species ..
        " genome " .. pool.cur_genome .. " neurons: " .. genome.network.num_neurons ..
        " genes: " .. #genome.genes ..  " fitness: " .. genome.fitness ..
        " score: " .. g.score)

    savestate.load(INIT_STATE);
    recent_games = {}
    survived_incomings = 0
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

  emu.frameadvance();
end
