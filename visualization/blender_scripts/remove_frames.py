import bpy

# Iterate over all objects in the scene
for obj in bpy.data.objects:
    # Check if "frame" is in the object's name
    if "frame" in obj.name:
        # Delete the object
        bpy.data.objects.remove(obj, do_unlink=True)