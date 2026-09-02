from __future__ import annotations

from datetime import datetime
from pathlib import Path

from manual_editing import EditingHistoryController, load_edit_batch
from manual_editing_theme import apply_dark_theme


def _default_checkpoint_path(controller: EditingHistoryController) -> Path:
    source_dir = controller.source_manifest.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return source_dir.parent / (
        f"{source_dir.name}_next_from_edit_{controller.position}_{stamp}")


def launch_history_controls(source_path: Path) -> Path | None:
    """Show checkpoint controls and return the source chosen for later edits."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    controller = EditingHistoryController(source_path)
    result: dict[str, Path | None] = {"path": None}
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "the history controls require a graphical desktop session") from error

    apply_dark_theme(root)

    root.title("Motion manual-editing history")
    root.geometry("1040x650")
    root.minsize(820, 540)

    outer = ttk.Frame(root, padding=14)
    outer.pack(fill="both", expand=True)
    ttk.Label(
        outer, text="Edit history", font=("TkDefaultFont", 13, "bold")
    ).pack(anchor="w")
    ttk.Label(
        outer,
        text=("Choose any recorded edit, restore its exact state, then continue "
              "from it without deleting the later branch."),
    ).pack(anchor="w", pady=(2, 10))

    table_frame = ttk.Frame(outer)
    table_frame.pack(fill="both", expand=True)
    columns = ("action", "scope", "user", "time", "pixels")
    tree = ttk.Treeview(
        table_frame, columns=columns, show="tree headings", selectmode="browse")
    tree.heading("#0", text="Step")
    tree.heading("action", text="Edit")
    tree.heading("scope", text="Scope")
    tree.heading("user", text="User")
    tree.heading("time", text="Recorded")
    tree.heading("pixels", text="Changed pixels")
    tree.column("#0", width=70, stretch=False, anchor="center")
    tree.column("action", width=250)
    tree.column("scope", width=170)
    tree.column("user", width=110)
    tree.column("time", width=165)
    tree.column("pixels", width=100, stretch=False, anchor="e")
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for checkpoint in controller.checkpoints:
        position = checkpoint.operation_count
        tree.insert(
            "", "end", iid=str(position), text=str(position), values=(
                checkpoint.action,
                checkpoint.scope,
                checkpoint.user,
                checkpoint.created_at_utc,
                f"{checkpoint.changed_pixels:,}",
            ))

    status = tk.StringVar()
    output_path = tk.StringVar(value=str(_default_checkpoint_path(controller)))

    details = ttk.Frame(outer, padding=(0, 10, 0, 0))
    details.pack(fill="x")
    ttk.Label(details, textvariable=status).pack(anchor="w")
    path_row = ttk.Frame(details)
    path_row.pack(fill="x", pady=(8, 0))
    ttk.Label(path_row, text="New session folder:").pack(side="left")
    output_entry = ttk.Entry(path_row, textvariable=output_path)
    output_entry.pack(side="left", fill="x", expand=True, padx=(8, 6))

    def browse_output() -> None:
        parent = filedialog.askdirectory(
            title="Choose where to save the new branch checkpoint",
            initialdir=str(controller.source_manifest.parent.parent))
        if parent:
            output_path.set(str(Path(parent) / _default_checkpoint_path(
                controller).name))

    browse_button = ttk.Button(
        path_row, text="Browse…", command=browse_output)
    browse_button.pack(side="right")

    remove_row = ttk.Frame(details)
    remove_row.pack(fill="x", pady=(8, 0))
    ttk.Label(
        remove_row, text="Identities to remove (comma-separated):"
    ).pack(side="left")
    identity_value = tk.StringVar()
    identity_entry = ttk.Entry(
        remove_row, textvariable=identity_value, width=12)
    identity_entry.pack(side="left", padx=(8, 6))

    filter_row = ttk.Frame(details)
    filter_row.pack(fill="x", pady=(8, 0))
    ttk.Label(
        filter_row,
        text="Filter by area, duration, gaps, movement, quality, or contacts:",
    ).pack(side="left")

    def filtered_session_saved(path: Path) -> None:
        result["path"] = path
        root.destroy()

    def open_filter_controls() -> None:
        from manual_editing_filter_ui import launch_filter_controls
        launch_filter_controls(
            root, controller, output_path.get, filtered_session_saved)

    ttk.Button(
        filter_row, text="Apply operation by track filters…",
        command=open_filter_controls,
    ).pack(side="left", padx=(8, 0))

    expand_row = ttk.Frame(details)
    expand_row.pack(fill="x", pady=(8, 0))
    ttk.Label(
        expand_row, text="Identities to expand (comma-separated):"
    ).pack(side="left")
    expand_identity_value = tk.StringVar()
    ttk.Entry(
        expand_row, textvariable=expand_identity_value, width=12
    ).pack(side="left", padx=(8, 12))
    ttk.Label(expand_row, text="Radius:").pack(side="left")
    expand_radius_value = tk.StringVar(value="1")
    ttk.Entry(
        expand_row, textvariable=expand_radius_value, width=8
    ).pack(side="left", padx=(6, 6))
    expand_unit_value = tk.StringVar(value="calibrated")
    ttk.Combobox(
        expand_row, textvariable=expand_unit_value,
        values=("calibrated", "pixel"), state="readonly", width=11,
    ).pack(side="left", padx=(0, 6))

    batch_row = ttk.Frame(details)
    batch_row.pack(fill="x", pady=(8, 0))
    ttk.Label(batch_row, text="Edit-batch JSON file:").pack(side="left")
    batch_path = tk.StringVar()
    ttk.Entry(batch_row, textvariable=batch_path).pack(
        side="left", fill="x", expand=True, padx=(8, 6))

    def browse_batch() -> None:
        selected = filedialog.askopenfilename(
            title="Choose an edit-batch JSON file",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if selected:
            batch_path.set(selected)

    ttk.Button(
        batch_row, text="Browse…", command=browse_batch
    ).pack(side="right")

    buttons = ttk.Frame(outer, padding=(0, 12, 0, 0))
    buttons.pack(fill="x")

    def show_current() -> None:
        tree.selection_set(str(controller.position))
        tree.focus(str(controller.position))
        tree.see(str(controller.position))
        status.set(
            f"Current state: checkpoint {controller.position} of "
            f"{len(controller.operations)}")
        undo_button.configure(
            state="normal" if controller.can_undo else "disabled")
        redo_button.configure(
            state="normal" if controller.can_redo else "disabled")
        earlier = controller.position < len(controller.operations)
        continue_button.configure(
            text="Save branch and continue" if earlier else "Continue from here")

    def restore_selected() -> None:
        selected = tree.selection()
        if selected:
            controller.restore(int(selected[0]))
            output_path.set(str(_default_checkpoint_path(controller)))
            show_current()

    def undo() -> None:
        if controller.can_undo:
            controller.undo()
            output_path.set(str(_default_checkpoint_path(controller)))
            show_current()

    def redo() -> None:
        if controller.can_redo:
            controller.redo()
            output_path.set(str(_default_checkpoint_path(controller)))
            show_current()

    def continue_from_here() -> None:
        try:
            target = None
            if controller.position < len(controller.operations):
                raw_target = output_path.get().strip()
                if not raw_target:
                    raise ValueError("choose a folder for the new branch checkpoint")
                target = Path(raw_target)
            result["path"] = controller.continue_from_here(target)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot continue from checkpoint", str(error))
            return
        root.destroy()

    def remove_identity() -> None:
        try:
            identities = [
                int(value) for value in identity_value.get().replace(",", " ").split()
            ]
            if not identities:
                raise ValueError("enter at least one identity to remove")
            raw_target = output_path.get().strip()
            if not raw_target:
                raise ValueError("choose a folder for the new editing session")
            result["path"] = controller.remove_identities(
                identities, Path(raw_target))
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot remove identities", str(error))
            return
        root.destroy()

    ttk.Button(
        remove_row, text="Remove identities and save batch",
        command=remove_identity,
    ).pack(side="left")

    def expand_identities() -> None:
        try:
            identities = [
                int(value)
                for value in expand_identity_value.get().replace(",", " ").split()
            ]
            if not identities:
                raise ValueError("enter at least one identity to expand")
            radius = float(expand_radius_value.get())
            raw_target = output_path.get().strip()
            if not raw_target:
                raise ValueError("choose a folder for the new editing session")
            result["path"] = controller.expand_identities(
                identities, radius, Path(raw_target), expand_unit_value.get())
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot expand identities", str(error))
            return
        root.destroy()

    ttk.Button(
        expand_row, text="Expand identities and save batch",
        command=expand_identities,
    ).pack(side="left")

    def apply_batch() -> None:
        try:
            raw_batch = batch_path.get().strip()
            raw_target = output_path.get().strip()
            if not raw_batch:
                raise ValueError("choose an edit-batch JSON file")
            if not raw_target:
                raise ValueError("choose a folder for the new editing session")
            operations = load_edit_batch(Path(raw_batch))
            result["path"] = controller.apply_batch(
                operations, Path(raw_target))
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot apply edit batch", str(error))
            return
        root.destroy()

    ttk.Button(
        batch_row, text="Apply batch and save", command=apply_batch
    ).pack(side="right", padx=(6, 0))

    undo_button = ttk.Button(buttons, text="Undo", command=undo)
    undo_button.pack(side="left")
    redo_button = ttk.Button(buttons, text="Redo", command=redo)
    redo_button.pack(side="left", padx=(6, 0))
    ttk.Button(
        buttons, text="Restore selected", command=restore_selected
    ).pack(side="left", padx=(18, 0))
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right")
    continue_button = ttk.Button(
        buttons, text="Continue from here", command=continue_from_here)
    continue_button.pack(side="right", padx=(0, 6))

    tree.bind("<Double-1>", lambda _event: restore_selected())
    root.bind("<Control-z>", lambda _event: undo())
    root.bind("<Control-y>", lambda _event: redo())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    show_current()
    root.mainloop()
    return result["path"]
