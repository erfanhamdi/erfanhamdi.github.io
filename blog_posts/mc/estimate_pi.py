import numpy as np
import matplotlib.pyplot as plt

no_of_points = 1_000
x = np.random.uniform(-1, 1, no_of_points)
y = np.random.uniform(-1, 1, no_of_points)
inside = x**2 + y**2 <= 1
pi = 4 * inside.sum() / no_of_points
print(pi)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.add_patch(plt.Circle((0, 0), 1, color='r', fill=False))
ax.add_patch(plt.Rectangle((-1, -1), 2, 2, color='b', fill=False))
# write the radius of the circle on an arrow inside it
# ax.annotate('r=1', xy=(0, 0), xytext=(0.5, 0.5),
#             arrowprops=dict(facecolor='black', shrink=0.05))

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off') 
plt.savefig('blog_posts/mc/square_circle.png')
ax.scatter(x[inside], y[inside], color='r', s=1)
ax.scatter(x[~inside], y[~inside], color='b', s=1)
plt.savefig('blog_posts/mc/darts_flying.png')
plt.show()
# turn off the axis

