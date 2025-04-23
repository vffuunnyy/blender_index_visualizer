import contextlib

from collections import namedtuple

import blf
import bmesh
import bpy
import gpu
import mathutils

from bpy.props import *
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader


PERSISTENT_VERT_ID_LAYER = "persistent_vert_id"
PERSISTENT_EDGE_ID_LAYER = "persistent_edge_id"
PERSISTENT_FACE_ID_LAYER = "persistent_face_id"

SELECTION_VERT_LAYER = "iv_vert_selected"
SELECTION_EDGE_LAYER = "iv_edge_selected"
SELECTION_FACE_LAYER = "iv_face_selected"

NO_ID_VALUE = -1

Rect = namedtuple("Rect", "x0 y0 x1 y1")

addon_keymaps = []

mode_change_handler = None


def register_mode_change_handler():
    """Регистрирует обработчик события смены режима"""
    global mode_change_handler
    if mode_change_handler is None:
        mode_change_handler = bpy.app.handlers.depsgraph_update_post.append(check_mode_change)
        print("Mode change handler registered")


def unregister_mode_change_handler():
    """Удаляет обработчик события смены режима"""
    global mode_change_handler
    if mode_change_handler is not None:
        with contextlib.suppress(ValueError):
            bpy.app.handlers.depsgraph_update_post.remove(mode_change_handler)
        mode_change_handler = None
        print("Mode change handler unregistered")


prev_mode = None


def check_mode_change(scene, depsgraph):
    global prev_mode

    active_obj = bpy.context.view_layer.objects.active
    if active_obj is None:
        return

    current_mode = active_obj.mode

    if prev_mode != current_mode:
        if prev_mode == "EDIT" and current_mode != "EDIT":
            update_selection_state(active_obj)

        prev_mode = current_mode


class IVProperties(bpy.types.PropertyGroup):
    running: BoolProperty(
        name="Работает", description="Включена ли визуализация индексов?", default=False
    )
    show_verts: BoolProperty(name="Вершины", description="Показывать индексы вершин", default=True)
    show_edges: BoolProperty(name="Рёбра", description="Показывать индексы рёбер", default=True)
    show_faces: BoolProperty(
        name="Грани", description="Показывать индексы граней (плоскостей)", default=True
    )


def get_canvas(_: bpy.context, pos: mathutils.Vector, ch_count: int, font_size: int) -> Rect:
    width = ch_count * font_size * 1.0
    height = font_size * 1.5

    x0 = int(pos.x - width * 0.5)
    y0 = int(pos.y - height * 0.5)
    x1 = int(pos.x + width * 0.5)
    y1 = int(pos.y + height * 0.5)
    return Rect(x0, y0, x1, y1)


class IVRenderer(bpy.types.Operator):
    bl_idname = "view3d.iv_renderer"
    bl_label = "Index renderer"

    _handle = None

    @staticmethod
    def handle_add(context: bpy.context) -> None:
        if IVRenderer._handle is None:
            IVRenderer._handle = bpy.types.SpaceView3D.draw_handler_add(
                IVRenderer._draw_callback, (context,), "WINDOW", "POST_PIXEL"
            )

    @staticmethod
    def handle_remove(context: bpy.context) -> None:
        if IVRenderer._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(IVRenderer._handle, "WINDOW")
            IVRenderer._handle = None

    @staticmethod
    def _draw_callback(context: bpy.context) -> None:
        props = context.scene.iv_props
        if not props.running:
            return

        if context.object is None:
            return

        obj = context.active_object
        world_mat = obj.matrix_world

        vert_data = []
        edge_data = []
        face_data = []

        if obj.type != "MESH":
            return

        mesh = obj.data
        prev_mode = obj.mode

        need_mode_restore = False

        if prev_mode != "EDIT":
            selected_objects = [o for o in bpy.context.selected_objects]
            active_object = bpy.context.view_layer.objects.active

            bpy.ops.object.mode_set(mode="EDIT")
            need_mode_restore = True

        bm = bmesh.from_edit_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        try:
            vert_id_layer = bm.verts.layers.int.get(PERSISTENT_VERT_ID_LAYER)
            edge_id_layer = bm.edges.layers.int.get(PERSISTENT_EDGE_ID_LAYER)
            face_id_layer = bm.faces.layers.int.get(PERSISTENT_FACE_ID_LAYER)

            vert_sel_layer = bm.verts.layers.int.get(SELECTION_VERT_LAYER)
            edge_sel_layer = bm.edges.layers.int.get(SELECTION_EDGE_LAYER)
            face_sel_layer = bm.faces.layers.int.get(SELECTION_FACE_LAYER)

            if prev_mode == "EDIT":
                if props.show_verts and vert_id_layer is not None:
                    for v in bm.verts:
                        if v.select:
                            try:
                                persistent_id = v[vert_id_layer]
                                if persistent_id > 0:
                                    vert_data.append((persistent_id, world_mat @ v.co))
                            except Exception as e:
                                print(f"Err vert ID {v.index}: {e}")

                if props.show_edges and edge_id_layer is not None:
                    for e in bm.edges:
                        if e.select:
                            try:
                                persistent_id = e[edge_id_layer]
                                if persistent_id > 0:
                                    center_coord = world_mat @ ((e.verts[0].co + e.verts[1].co) / 2)
                                    edge_data.append((persistent_id, center_coord))
                            except Exception as e:
                                print(f"Err edge ID {e.index}: {e}")

                if props.show_faces and face_id_layer is not None:
                    for f in bm.faces:
                        if f.select:
                            try:
                                persistent_id = f[face_id_layer]
                                if persistent_id > 0:
                                    center_coord = world_mat @ f.calc_center_median()
                                    face_data.append((persistent_id, center_coord))
                            except Exception as e:
                                print(f"Error reading persistent ID for face {f.index}: {e}")
            else:
                if props.show_verts and vert_id_layer is not None and vert_sel_layer is not None:
                    for v in bm.verts:
                        try:
                            was_selected = v[vert_sel_layer] == 1
                            if was_selected:
                                persistent_id = v[vert_id_layer]
                                if persistent_id > 0:
                                    vert_data.append((persistent_id, world_mat @ v.co))
                        except Exception as e:
                            print(f"Err vert ID {v.index}: {e}")

                if props.show_edges and edge_id_layer is not None and edge_sel_layer is not None:
                    for e in bm.edges:
                        try:
                            was_selected = e[edge_sel_layer] == 1
                            if was_selected:
                                persistent_id = e[edge_id_layer]
                                if persistent_id > 0:
                                    center_coord = world_mat @ ((e.verts[0].co + e.verts[1].co) / 2)
                                    edge_data.append((persistent_id, center_coord))
                        except Exception as e:
                            print(f"Err edge ID {e.index}: {e}")

                if props.show_faces and face_id_layer is not None and face_sel_layer is not None:
                    for f in bm.faces:
                        try:
                            was_selected = f[face_sel_layer] == 1
                            if was_selected:
                                persistent_id = f[face_id_layer]
                                if persistent_id > 0:
                                    center_coord = world_mat @ f.calc_center_median()
                                    face_data.append((persistent_id, center_coord))
                        except Exception as e:
                            print(f"Error reading persistent ID for face {f.index}: {e}")
        finally:
            bm.free()

            if need_mode_restore:
                bpy.ops.object.mode_set(mode=prev_mode)

                bpy.context.view_layer.objects.active = active_object
                for obj in bpy.context.selected_objects:
                    obj.select_set(False)
                for obj in selected_objects:
                    obj.select_set(True)

        if vert_data:
            IVRenderer._render_data(context, vert_data)
        if edge_data:
            IVRenderer._render_data(context, edge_data)
        if face_data:
            IVRenderer._render_data(context, face_data)

    @staticmethod
    def _render_data(context: bpy.context, data: list) -> None:
        for item in data:
            IVRenderer._render_single(context, item)

    @staticmethod
    def _render_single(context: bpy.context, data: tuple) -> None:
        index_str = str(data[0])
        position = data[1]
        sc = context.scene

        region = context.region
        region_3d = context.space_data.region_3d
        loc_2d = view3d_utils.location_3d_to_region_2d(region, region_3d, position)
        if not loc_2d:
            return

        rect = get_canvas(context, loc_2d, len(index_str), sc.iv_font_size)
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")

        vertices = ((rect.x0, rect.y0), (rect.x0, rect.y1), (rect.x1, rect.y1), (rect.x1, rect.y0))
        indices = ((0, 1, 2), (2, 3, 0))
        batch = batch_for_shader(shader, "TRIS", {"pos": vertices}, indices=indices)
        shader.bind()
        shader.uniform_float("color", (*sc.iv_box_color[:3], sc.iv_box_color[3]))
        batch.draw(shader)

        blf.size(0, sc.iv_font_size)
        blf.position(0, rect.x0 + 5, rect.y0 + 5, 0)
        blf.color(0, *sc.iv_text_color)
        blf.draw(0, index_str)


class IVOperator(bpy.types.Operator):
    bl_idname = "view3d.iv_op"
    bl_label = "Визуализатор индексов"
    bl_description = "Включить/выключить визуализацию индексов"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.context):
        props = context.scene.iv_props
        if props.running:
            IVRenderer.handle_remove(context)
            unregister_mode_change_handler()
        else:
            IVRenderer.handle_add(context)
            register_mode_change_handler()
            active_obj = context.active_object
            if active_obj and active_obj.mode == "EDIT":
                update_selection_state(active_obj)
        props.running = not props.running
        context.region.tag_redraw()
        return {"FINISHED"}


class IV_PT_Panel(bpy.types.Panel):
    bl_label = "Визуализатор индексов"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Инструменты"

    def draw(self, context: bpy.context) -> None:
        layout = self.layout
        props = context.scene.iv_props

        if props.running:
            layout.operator(IVOperator.bl_idname, text="Остановить", icon="PAUSE")
            layout.prop(props, "show_verts")
            layout.prop(props, "show_edges")
            layout.prop(props, "show_faces")
            layout.separator()
            layout.label(text="Присвоить постоянные ID:")
            row = layout.row()
            row.operator(AssignPersistentVertIDsOperator.bl_idname, text="Вершины")
            row.operator(AssignPersistentEdgeIDsOperator.bl_idname, text="Рёбра")
            row.operator(AssignPersistentFaceIDsOperator.bl_idname, text="Грани")

            layout.label(text="Удалить ID:")
            row = layout.row()
            row.operator(ClearVertIDsOperator.bl_idname, text="Вершины")
            row.operator(ClearEdgeIDsOperator.bl_idname, text="Рёбра")
            row.operator(ClearFaceIDsOperator.bl_idname, text="Грани")

            layout.separator()
            layout.prop(context.scene, "iv_box_color", text="Цвет фона")
            layout.prop(context.scene, "iv_text_color", text="Цвет текста")
            layout.prop(context.scene, "iv_font_size", text="Размер шрифта")
        else:
            layout.operator(IVOperator.bl_idname, text="Запустить", icon="PLAY")


def init_properties() -> None:
    bpy.types.Scene.iv_props = PointerProperty(type=IVProperties)
    bpy.types.Scene.iv_box_color = FloatVectorProperty(
        name="Цвет фона", subtype="COLOR", size=4, default=(0.0, 0.0, 0.0, 0.7), min=0.0, max=1.0
    )
    bpy.types.Scene.iv_text_color = FloatVectorProperty(
        name="Цвет текста", subtype="COLOR", size=4, default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0
    )
    bpy.types.Scene.iv_font_size = IntProperty(name="Размер шрифта", default=14, min=10, max=50)


def clear_properties() -> None:
    del bpy.types.Scene.iv_props
    del bpy.types.Scene.iv_box_color
    del bpy.types.Scene.iv_text_color
    del bpy.types.Scene.iv_font_size


def register() -> None:
    bpy.utils.register_class(IVProperties)
    bpy.utils.register_class(IVRenderer)
    bpy.utils.register_class(IVOperator)
    bpy.utils.register_class(IV_PT_Panel)
    bpy.utils.register_class(AssignPersistentFaceIDsOperator)
    bpy.utils.register_class(AssignPersistentVertIDsOperator)
    bpy.utils.register_class(AssignPersistentEdgeIDsOperator)
    bpy.utils.register_class(ClearVertIDsOperator)
    bpy.utils.register_class(ClearEdgeIDsOperator)
    bpy.utils.register_class(ClearFaceIDsOperator)
    bpy.utils.register_class(ModeChangeHandler)
    init_properties()

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        kmi = km.keymap_items.new(IVOperator.bl_idname, "I", "PRESS", ctrl=True, shift=True)
        addon_keymaps.append((km, kmi))


def unregister() -> None:
    unregister_mode_change_handler()

    if IVRenderer._handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(IVRenderer._handle, "WINDOW")
        IVRenderer._handle = None

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    clear_properties()
    bpy.utils.unregister_class(IV_PT_Panel)
    bpy.utils.unregister_class(IVOperator)
    bpy.utils.unregister_class(IVRenderer)
    bpy.utils.unregister_class(IVProperties)
    bpy.utils.unregister_class(AssignPersistentFaceIDsOperator)
    bpy.utils.unregister_class(AssignPersistentVertIDsOperator)
    bpy.utils.unregister_class(AssignPersistentEdgeIDsOperator)
    bpy.utils.unregister_class(ClearVertIDsOperator)
    bpy.utils.unregister_class(ClearEdgeIDsOperator)
    bpy.utils.unregister_class(ClearFaceIDsOperator)
    bpy.utils.unregister_class(ModeChangeHandler)


class AssignPersistentFaceIDsOperator(bpy.types.Operator):
    bl_idname = "mesh.assign_persistent_face_ids"
    bl_label = "Присвоить постоянные ID граням"
    bl_description = "Присваивает уникальные постоянные ID выделенным граням, у которых их ещё нет"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.context) -> bool:
        return (
            context.active_object is not None
            and context.active_object.mode == "EDIT"
            and context.active_object.type == "MESH"
        )

    def execute(self, context: bpy.context):
        obj = context.active_object
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table()
        print("--- AssignPersistentFaceIDsOperator ---")

        face_id_layer = bm.faces.layers.int.get(PERSISTENT_FACE_ID_LAYER)
        initialized_layer = False
        if face_id_layer is None:
            print(f"Слой '{PERSISTENT_FACE_ID_LAYER}' не найден. Создаю и инициализирую.")
            face_id_layer = bm.faces.layers.int.new(PERSISTENT_FACE_ID_LAYER)
            for f in bm.faces:
                try:
                    f[face_id_layer] = NO_ID_VALUE
                except Exception as e:
                    print(f"Ошибка инициализации грани {f.index}: {e}")
            initialized_layer = True
        else:
            print(f"Слой '{PERSISTENT_FACE_ID_LAYER}' найден.")

        current_max_id = 0
        print("Сканирую существующие ID:")
        face_ids_read = {}
        for f in bm.faces:
            try:
                face_id = f[face_id_layer]
                face_ids_read[f.index] = face_id
                if face_id > 0:
                    current_max_id = max(current_max_id, face_id)
            except Exception as e:
                print(f"Ошибка чтения ID грани {f.index}: {e}")
                face_ids_read[f.index] = f"Ошибка: {e}"

        print(f"Прочитанные ID: {face_ids_read}")
        print(f"Максимальный существующий ID: {current_max_id}")

        ids_assigned_this_run = {}
        next_id_to_assign = current_max_id + 1 if current_max_id > 0 else 1
        print(f"Следующий ID для присвоения: {next_id_to_assign}")
        print("Присваиваю ID выделенным граням (если не назначен):")
        for f in bm.faces:
            if f.select:
                try:
                    current_id = f[face_id_layer]
                    if current_id <= 0:
                        print(
                            f"  Присваиваю ID {next_id_to_assign} выделенной грани {f.index} (текущий ID: {current_id})"
                        )
                        f[face_id_layer] = next_id_to_assign
                        ids_assigned_this_run[f.index] = next_id_to_assign
                        next_id_to_assign += 1
                    else:
                        print(f"  Грани {f.index} уже присвоен ID {current_id}. Пропускаю.")

                except Exception as e:
                    print(f"Ошибка обработки выделенной грани {f.index}: {e}")

        if ids_assigned_this_run or initialized_layer:
            print("Обновляю меш...")
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            context.area.tag_redraw()
            print("Меш обновлён и перерисован.")
        else:
            print("Новых ID не присвоено и слой не инициализирован. Обновление не требуется.")

        bm.free()

        update_selection_state(obj)

        report_message = f"ID обработаны. Присвоено: {len(ids_assigned_this_run)}. Следующий: {next_id_to_assign}"
        self.report({"INFO"}, report_message)
        print(report_message)
        print("--- Оператор завершён ---")
        return {"FINISHED"}


class AssignPersistentVertIDsOperator(bpy.types.Operator):
    bl_idname = "mesh.assign_persistent_vert_ids"
    bl_label = "Присвоить постоянные ID вершинам"
    bl_description = (
        "Присваивает уникальные постоянные ID выделенным вершинам, у которых их ещё нет"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.context) -> bool:
        return (
            context.active_object is not None
            and context.active_object.mode == "EDIT"
            and context.active_object.type == "MESH"
        )

    def execute(self, context: bpy.context):
        obj = context.active_object
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        bm.verts.ensure_lookup_table()
        print("--- AssignPersistentVertIDsOperator ---")

        id_layer = bm.verts.layers.int.get(PERSISTENT_VERT_ID_LAYER)
        initialized_layer = False
        if id_layer is None:
            print(f"Слой '{PERSISTENT_VERT_ID_LAYER}' не найден. Создаю...")
            id_layer = bm.verts.layers.int.new(PERSISTENT_VERT_ID_LAYER)
            for v in bm.verts:
                try:
                    v[id_layer] = NO_ID_VALUE
                except Exception as e:
                    print(f"Ошибка инициализации вершины {v.index}: {e}")
            initialized_layer = True
        else:
            print(f"Слой '{PERSISTENT_VERT_ID_LAYER}' найден.")

        current_max_id = 0
        for v in bm.verts:
            try:
                vert_id = v[id_layer]
                if vert_id > 0:
                    current_max_id = max(current_max_id, vert_id)
            except Exception as e:
                print(f"Ошибка чтения ID вершины {v.index}: {e}")
        print(f"Максимальный существующий ID вершины: {current_max_id}")

        ids_assigned_count = 0
        next_id_to_assign = current_max_id + 1 if current_max_id > 0 else 1
        print(f"Следующий ID для вершины: {next_id_to_assign}")
        for v in bm.verts:
            if v.select:
                try:
                    if v[id_layer] <= 0:
                        print(f"  Присваиваю ID {next_id_to_assign} выделенной вершине {v.index}")
                        v[id_layer] = next_id_to_assign
                        next_id_to_assign += 1
                        ids_assigned_count += 1
                    else:
                        print(f"  Вершине {v.index} уже присвоен ID {v[id_layer]}. Пропускаю.")
                except Exception as e:
                    print(f"Ошибка обработки вершины {v.index}: {e}")

        if ids_assigned_count > 0 or initialized_layer:
            print("Обновляю меш...")
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            context.area.tag_redraw()
            print("Меш обновлён.")
        else:
            print("Новых ID для вершин не присвоено.")

        bm.free()

        update_selection_state(obj)

        self.report(
            {"INFO"},
            f"ID вершин обработаны. Присвоено: {ids_assigned_count}. Следующий: {next_id_to_assign}",
        )
        print("--- Vert Operator Finished ---")
        return {"FINISHED"}


class AssignPersistentEdgeIDsOperator(bpy.types.Operator):
    bl_idname = "mesh.assign_persistent_edge_ids"
    bl_label = "Присвоить постоянные ID рёбрам"
    bl_description = "Присваивает уникальные постоянные ID выделенным рёбрам, у которых их ещё нет"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.context) -> bool:
        return (
            context.active_object is not None
            and context.active_object.mode == "EDIT"
            and context.active_object.type == "MESH"
        )

    def execute(self, context: bpy.context):
        obj = context.active_object
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        bm.edges.ensure_lookup_table()
        print("--- AssignPersistentEdgeIDsOperator ---")

        id_layer = bm.edges.layers.int.get(PERSISTENT_EDGE_ID_LAYER)
        initialized_layer = False
        if id_layer is None:
            print(f"Слой '{PERSISTENT_EDGE_ID_LAYER}' не найден. Создаю...")
            id_layer = bm.edges.layers.int.new(PERSISTENT_EDGE_ID_LAYER)
            for e in bm.edges:
                try:
                    e[id_layer] = NO_ID_VALUE
                except Exception as err:
                    print(f"Ошибка инициализации ребра {e.index}: {err}")
            initialized_layer = True
        else:
            print(f"Слой '{PERSISTENT_EDGE_ID_LAYER}' найден.")

        current_max_id = 0
        for e in bm.edges:
            try:
                edge_id = e[id_layer]
                if edge_id > 0:
                    current_max_id = max(current_max_id, edge_id)
            except Exception as err:
                print(f"Ошибка чтения ID ребра {e.index}: {err}")
        print(f"Максимальный существующий ID ребра: {current_max_id}")

        ids_assigned_count = 0
        next_id_to_assign = current_max_id + 1 if current_max_id > 0 else 1
        print(f"Следующий ID для ребра: {next_id_to_assign}")
        for e in bm.edges:
            if e.select:
                try:
                    if e[id_layer] <= 0:
                        print(f"  Присваиваю ID {next_id_to_assign} выделенному ребру {e.index}")
                        e[id_layer] = next_id_to_assign
                        next_id_to_assign += 1
                        ids_assigned_count += 1
                    else:
                        print(f"  Рёбру {e.index} уже присвоен ID {e[id_layer]}. Пропускаю.")
                except Exception as err:
                    print(f"Ошибка обработки ребра {e.index}: {err}")

        if ids_assigned_count > 0 or initialized_layer:
            print("Обновляю меш...")
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            context.area.tag_redraw()
            print("Меш обновлён.")
        else:
            print("Новых ID для рёбер не присвоено.")

        bm.free()

        update_selection_state(obj)

        self.report(
            {"INFO"},
            f"ID рёбер обработаны. Присвоено: {ids_assigned_count}. Следующий: {next_id_to_assign}",
        )
        print("--- Edge Operator Finished ---")
        return {"FINISHED"}


class ClearVertIDsOperator(bpy.types.Operator):
    bl_idname = "mesh.clear_persistent_vert_ids"
    bl_label = "Удалить ID вершин"
    bl_description = "Удаляет постоянные ID у выделенных вершин"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.context) -> bool:
        return (
            context.active_object is not None
            and context.active_object.mode == "EDIT"
            and context.active_object.type == "MESH"
        )

    def execute(self, context: bpy.context):
        obj = context.active_object
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        bm.verts.ensure_lookup_table()
        print("--- ClearVertIDsOperator ---")

        id_layer = bm.verts.layers.int.get(PERSISTENT_VERT_ID_LAYER)
        if id_layer is None:
            self.report({"INFO"}, "Слой с ID вершин не найден")
            bm.free()
            return {"CANCELLED"}

        cleared_count = 0
        for v in bm.verts:
            if v.select:
                try:
                    if v[id_layer] > 0:
                        v[id_layer] = NO_ID_VALUE
                        cleared_count += 1
                except Exception as e:
                    print(f"Ошибка при очистке ID вершины {v.index}: {e}")

        if cleared_count > 0:
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            context.area.tag_redraw()

        bm.free()

        update_selection_state(obj)

        self.report({"INFO"}, f"Удалено ID у {cleared_count} вершин")
        return {"FINISHED"}


class ClearEdgeIDsOperator(bpy.types.Operator):
    bl_idname = "mesh.clear_persistent_edge_ids"
    bl_label = "Удалить ID рёбер"
    bl_description = "Удаляет постоянные ID у выделенных рёбер"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.context) -> bool:
        return (
            context.active_object is not None
            and context.active_object.mode == "EDIT"
            and context.active_object.type == "MESH"
        )

    def execute(self, context: bpy.context):
        obj = context.active_object
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        bm.edges.ensure_lookup_table()
        print("--- ClearEdgeIDsOperator ---")

        id_layer = bm.edges.layers.int.get(PERSISTENT_EDGE_ID_LAYER)
        if id_layer is None:
            self.report({"INFO"}, "Слой с ID рёбер не найден")
            bm.free()
            return {"CANCELLED"}

        cleared_count = 0
        for e in bm.edges:
            if e.select:
                try:
                    if e[id_layer] > 0:
                        e[id_layer] = NO_ID_VALUE
                        cleared_count += 1
                except Exception as e:
                    print(f"Ошибка при очистке ID ребра {e.index}: {e}")

        if cleared_count > 0:
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            context.area.tag_redraw()

        bm.free()

        update_selection_state(obj)

        self.report({"INFO"}, f"Удалено ID у {cleared_count} рёбер")
        return {"FINISHED"}


class ClearFaceIDsOperator(bpy.types.Operator):
    bl_idname = "mesh.clear_persistent_face_ids"
    bl_label = "Удалить ID граней"
    bl_description = "Удаляет постоянные ID у выделенных граней"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.context) -> bool:
        return (
            context.active_object is not None
            and context.active_object.mode == "EDIT"
            and context.active_object.type == "MESH"
        )

    def execute(self, context: bpy.context):
        obj = context.active_object
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table()
        print("--- ClearFaceIDsOperator ---")

        id_layer = bm.faces.layers.int.get(PERSISTENT_FACE_ID_LAYER)
        if id_layer is None:
            self.report({"INFO"}, "Слой с ID граней не найден")
            bm.free()
            return {"CANCELLED"}

        cleared_count = 0
        for f in bm.faces:
            if f.select:
                try:
                    if f[id_layer] > 0:
                        f[id_layer] = NO_ID_VALUE
                        cleared_count += 1
                except Exception as e:
                    print(f"Ошибка при очистке ID грани {f.index}: {e}")

        if cleared_count > 0:
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            context.area.tag_redraw()

        bm.free()

        update_selection_state(obj)

        self.report({"INFO"}, f"Удалено ID у {cleared_count} граней")
        return {"FINISHED"}


def update_selection_state(obj):
    """Обновляет состояние выделения в custom data layers"""
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return

    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    vert_sel_layer = bm.verts.layers.int.get(SELECTION_VERT_LAYER)
    if vert_sel_layer is None:
        vert_sel_layer = bm.verts.layers.int.new(SELECTION_VERT_LAYER)

    edge_sel_layer = bm.edges.layers.int.get(SELECTION_EDGE_LAYER)
    if edge_sel_layer is None:
        edge_sel_layer = bm.edges.layers.int.new(SELECTION_EDGE_LAYER)

    face_sel_layer = bm.faces.layers.int.get(SELECTION_FACE_LAYER)
    if face_sel_layer is None:
        face_sel_layer = bm.faces.layers.int.new(SELECTION_FACE_LAYER)

    for v in bm.verts:
        v[vert_sel_layer] = 1 if v.select else 0

    for e in bm.edges:
        e[edge_sel_layer] = 1 if e.select else 0

    for f in bm.faces:
        f[face_sel_layer] = 1 if f.select else 0

    bmesh.update_edit_mesh(mesh)
    bm.free()

    print("Состояние выделения обновлено")


class ModeChangeHandler(bpy.types.Operator):
    bl_idname = "object.iv_mode_change_handler"
    bl_label = "IV Mode Change Handler"
    bl_description = "Внутренний оператор для отслеживания изменения режима объекта"
    bl_options = {"INTERNAL"}

    prev_mode = None

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        active_obj = context.active_object

        if ModeChangeHandler.prev_mode == "EDIT" and active_obj.mode != "EDIT":
            update_selection_state(active_obj)

        ModeChangeHandler.prev_mode = active_obj.mode
        return {"FINISHED"}


if __name__ == "__main__":
    register()
