# inbetween
import bpy

yellow = (1, 0.905, 0.05, 1.0)
red = (0.9, 0.1, 0.1, 1.0)

# Create materials if they don't exist
yellow_material = bpy.data.materials.get("YellowMaterial")
if not yellow_material:
    yellow_material = bpy.data.materials.new(name="YellowMaterial")
    yellow_material.diffuse_color = yellow
    yellow_material.use_nodes = False

red_material = bpy.data.materials.get("RedMaterial")
if not red_material:
    red_material = bpy.data.materials.new(name="RedMaterial")
    red_material.diffuse_color = red
    red_material.use_nodes = False

objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and "frame" in obj.name]

for i, obj in enumerate(objects):
    # Determine material based on frame range
    if i < 49 or i >= len(objects) - 49:
        material = red_material
    else:
        material = yellow_material

    # Assign the material
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

    # Set visibility keyframes
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=1)
    obj.keyframe_insert(data_path="hide_render", frame=1)

    frame = i + 1
    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)

    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=frame + 1)
    obj.keyframe_insert(data_path="hide_render", frame=frame + 1)

print(f"Processed {len(objects)} objects with 'frame' in their names.")