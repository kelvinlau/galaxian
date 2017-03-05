from galaxian2 import *

#g = Point(27,214)
#es = [Point(28,184), Point(32,200), Point(35,216)]
#vs = [Point(-2,10), Point(-4,10), Point(-5,10)]
g = Point(151,214)
dx = 0
es = [Point(159,220)]
vs = [Point(2,4)]
print g, es, vs

g = Rect(g+Point(0,-5),6,16)
es = map(lambda e: Rect(e+Point(0,-3),0,4), es)
ses = copy.deepcopy(es)

print g, es
for i in xrange(1, 6):
  g.x += dx
  ies = []
  for e, se, v in zip(es, ses, vs):
    e.x = int(round(se.x + v.x * i / 5))
    e.y = int(round(se.y + v.y * i / 5))
    if intersected(e, g):
      ies.append(e)
  print g, es, ies
