from vector2 import Vector2

vec1 = Vector2(1, 3)
vec2 = Vector2(2, 4)
print(vec1)
print(vec2)
print(vec1 + vec2)
print(vec1 - vec2)

vec1 += vec2
vec2 -= vec1
print(vec1)
print(vec2)

print(-vec1)

print(vec1.xy)
