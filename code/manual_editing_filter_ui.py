from __future__ import annotations

from pathlib import Path
from typing import Callable

from manual_editing import EditingHistoryController
from manual_editing_theme import apply_dark_theme
from manual_editing_filters import (FILTERABLE_METRICS,
                                    build_track_filter_metrics,
                                    evaluate_track_filters,
                                    save_filtered_operation_session,
                                    validate_filter_set)


def launch_filter_controls(
        parent, controller: EditingHistoryController,
        output_path: Callable[[], str], on_saved: Callable[[Path], None]) -> None:
    """Build, preview, and apply track filters from the active checkpoint."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    window = tk.Toplevel(parent)
    apply_dark_theme(window)
    window.title("Apply an operation by track filters")
    window.geometry("1040x710")
    window.minsize(820, 520)
    window.transient(parent)
    window.grab_set()

    outer = ttk.Frame(window, padding=14)
    outer.pack(fill="both", expand=True)
    ttk.Label(
        outer, text="Apply an operation by track filters",
        font=("TkDefaultFont", 13, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        outer,
        text=("Build one rule set, preview the exact identities, then remove, "
              "delete over a frame range, or expand them as one reversible batch."),
    ).pack(anchor="w", pady=(2, 10))

    try:
        metrics, metadata = build_track_filter_metrics(
            controller.source_manifest, controller.position)
    except (OSError, ValueError) as error:
        messagebox.showerror("Cannot calculate track metrics", str(error), parent=window)
        window.destroy()
        return

    ttk.Label(
        outer,
        text=(f"Area: {metadata['area_unit']}    Distance: "
              f"{metadata['distance_unit']}    Speed: {metadata['speed_unit']}    "
              "Global pipeline confidence: unavailable; derived quality is not a probability"),
    ).pack(anchor="w", pady=(0, 10))

    operation_frame = ttk.LabelFrame(outer, text="Operation", padding=10)
    operation_frame.pack(fill="x", pady=(0, 10))
    action_value = tk.StringVar(value="Remove")
    radius_value = tk.StringVar(value="1")
    radius_unit_value = tk.StringVar(value="calibrated")
    start_frame_value = tk.StringVar(value="1")
    end_frame_value = tk.StringVar(value=str(len(controller.bundle.canonical_labels)))
    ttk.Label(operation_frame, text="Action").pack(side="left")
    action_box = ttk.Combobox(
        operation_frame, textvariable=action_value,
        values=("Remove", "Delete frames", "Expand"),
        state="readonly", width=13)
    action_box.pack(side="left", padx=(8, 18))
    ttk.Label(operation_frame, text="Expansion radius").pack(side="left")
    radius_entry = ttk.Entry(
        operation_frame, textvariable=radius_value, width=9, state="disabled")
    radius_entry.pack(side="left", padx=(8, 6))
    radius_unit_box = ttk.Combobox(
        operation_frame, textvariable=radius_unit_value,
        values=("calibrated", "pixel"), state="disabled", width=11)
    radius_unit_box.pack(side="left")
    ttk.Label(operation_frame, text="Frames").pack(side="left", padx=(12, 4))
    start_frame_entry = ttk.Entry(
        operation_frame, textvariable=start_frame_value, width=6,
        state="disabled")
    start_frame_entry.pack(side="left")
    ttk.Label(operation_frame, text="to").pack(side="left", padx=3)
    end_frame_entry = ttk.Entry(
        operation_frame, textvariable=end_frame_value, width=6,
        state="disabled")
    end_frame_entry.pack(side="left")

    builder = ttk.LabelFrame(outer, text="Add filter", padding=10)
    builder.pack(fill="x")
    metric_value = tk.StringVar(value="median_area")
    operator_value = tk.StringVar(value="lt")
    threshold_value = tk.StringVar()
    combine_value = tk.StringVar(value="all")

    ttk.Label(builder, text="Metric").grid(row=0, column=0, sticky="w")
    metric_box = ttk.Combobox(
        builder, textvariable=metric_value,
        values=sorted(FILTERABLE_METRICS), state="readonly", width=31)
    metric_box.grid(row=1, column=0, sticky="ew", padx=(0, 8))
    ttk.Label(builder, text="Operator").grid(row=0, column=1, sticky="w")
    ttk.Combobox(
        builder, textvariable=operator_value,
        values=("lt", "lte", "gt", "gte", "eq", "between", "outside"),
        state="readonly", width=12,
    ).grid(row=1, column=1, sticky="ew", padx=(0, 8))
    ttk.Label(builder, text="Value (two comma-separated for ranges)").grid(
        row=0, column=2, sticky="w")
    threshold_entry = ttk.Entry(builder, textvariable=threshold_value, width=25)
    threshold_entry.grid(row=1, column=2, sticky="ew", padx=(0, 8))
    ttk.Label(builder, text="Combine filters").grid(row=0, column=3, sticky="w")
    ttk.Combobox(
        builder, textvariable=combine_value, values=("all", "any"),
        state="readonly", width=10,
    ).grid(row=1, column=3, sticky="ew", padx=(0, 8))
    builder.columnconfigure(0, weight=2)
    builder.columnconfigure(2, weight=1)

    filters: list[dict] = []
    previewed = {"selected": None, "signature": None}

    body = ttk.Panedwindow(outer, orient="horizontal")
    body.pack(fill="both", expand=True, pady=(10, 0))
    filters_frame = ttk.LabelFrame(body, text="Filters", padding=6)
    results_frame = ttk.LabelFrame(body, text="Previewed identities", padding=6)
    body.add(filters_frame, weight=1)
    body.add(results_frame, weight=1)

    filter_tree = ttk.Treeview(
        filters_frame, columns=("metric", "operator", "value"),
        show="headings", selectmode="browse")
    for column, title, width in (
            ("metric", "Metric", 190), ("operator", "Operator", 80),
            ("value", "Value", 130)):
        filter_tree.heading(column, text=title)
        filter_tree.column(column, width=width)
    filter_tree.pack(fill="both", expand=True)

    result_tree = ttk.Treeview(
        results_frame, columns=("identity", "matched", "summary"),
        show="headings")
    result_tree.heading("identity", text="Identity")
    result_tree.heading("matched", text="Matched filters")
    result_tree.heading("summary", text="Selected metric values")
    result_tree.column("identity", width=70, stretch=False)
    result_tree.column("matched", width=130)
    result_tree.column("summary", width=290)
    result_tree.pack(fill="both", expand=True)
    status = tk.StringVar(value="Add at least one filter, then preview it.")
    ttk.Label(results_frame, textvariable=status).pack(anchor="w", pady=(6, 0))

    def invalidate_preview(*_args) -> None:
        previewed["selected"] = None
        previewed["signature"] = None
        apply_button.configure(state="disabled")
        status.set("Filters changed; preview again before applying.")

    combine_value.trace_add("write", invalidate_preview)
    radius_value.trace_add("write", invalidate_preview)
    radius_unit_value.trace_add("write", invalidate_preview)
    start_frame_value.trace_add("write", invalidate_preview)
    end_frame_value.trace_add("write", invalidate_preview)

    def operation_changed(*_args) -> None:
        expanding = action_value.get() == "Expand"
        interval = action_value.get() in {"Delete frames", "Expand"}
        radius_entry.configure(state="normal" if expanding else "disabled")
        radius_unit_box.configure(state="readonly" if expanding else "disabled")
        start_frame_entry.configure(state="normal" if interval else "disabled")
        end_frame_entry.configure(state="normal" if interval else "disabled")
        invalidate_preview()

    action_value.trace_add("write", operation_changed)

    def current_spec() -> dict:
        return validate_filter_set({
            "schema": "motion.manual-editing-filter-set",
            "schema_version": 1,
            "combine": combine_value.get(),
            "filters": filters,
        })

    def add_filter() -> None:
        try:
            raw = threshold_value.get().strip()
            operator = operator_value.get()
            value = ([float(part.strip()) for part in raw.split(",")]
                     if operator in {"between", "outside"} else float(raw))
            condition = validate_filter_set({
                "schema": "motion.manual-editing-filter-set",
                "schema_version": 1,
                "filters": [{
                    "metric": metric_value.get(),
                    "operator": operator,
                    "value": value,
                }],
            })["filters"][0]
        except ValueError as error:
            messagebox.showerror("Invalid filter", str(error), parent=window)
            return
        condition["name"] = f"filter_{len(filters) + 1}"
        filters.append(condition)
        filter_tree.insert(
            "", "end", iid=str(len(filters) - 1), values=(
                condition["metric"], condition["operator"], condition["value"]))
        threshold_value.set("")
        invalidate_preview()

    ttk.Button(builder, text="Add filter", command=add_filter).grid(
        row=1, column=4, sticky="ew")

    def remove_filter() -> None:
        selected = filter_tree.selection()
        if not selected:
            return
        remove_index = int(selected[0])
        filters.pop(remove_index)
        for item in filter_tree.get_children():
            filter_tree.delete(item)
        for index, condition in enumerate(filters):
            condition["name"] = f"filter_{index + 1}"
            filter_tree.insert("", "end", iid=str(index), values=(
                condition["metric"], condition["operator"], condition["value"]))
        invalidate_preview()

    ttk.Button(
        filters_frame, text="Remove selected filter", command=remove_filter
    ).pack(anchor="w", pady=(6, 0))

    def preview() -> None:
        try:
            spec = current_spec()
            matches = evaluate_track_filters(metrics, spec)
        except ValueError as error:
            messagebox.showerror("Cannot preview filters", str(error), parent=window)
            return
        for item in result_tree.get_children():
            result_tree.delete(item)
        selected_rows = matches.loc[matches.selected]
        selected_metrics = list(dict.fromkeys(
            condition["metric"] for condition in spec["filters"]))
        for row in selected_rows.itertuples(index=False):
            summary = ", ".join(
                f"{metric}={getattr(row, metric):.4g}"
                for metric in selected_metrics)
            result_tree.insert("", "end", values=(
                int(row.identity), row.matched_filters, summary))
        selected = selected_rows.identity.astype(int).tolist()
        previewed["selected"] = selected
        previewed["signature"] = spec
        status.set(
            f"{len(selected)} of {len(matches)} active identities selected.")
        apply_button.configure(state="normal" if selected else "disabled")

    controls = ttk.Frame(outer, padding=(0, 10, 0, 0))
    controls.pack(fill="x")
    ttk.Button(controls, text="Preview identities", command=preview).pack(side="left")
    ttk.Button(controls, text="Cancel", command=window.destroy).pack(side="right")

    def apply() -> None:
        if previewed["selected"] is None or previewed["signature"] is None:
            messagebox.showerror(
                "Preview required", "Preview the filters before applying them.",
                parent=window)
            return
        raw_output = output_path().strip()
        if not raw_output:
            messagebox.showerror(
                "Output required", "Choose a new session folder in the history window.",
                parent=window)
            return
        try:
            operation_type = {
                "Remove": "exclude_identity",
                "Delete frames": "delete_identity_interval",
                "Expand": "expand_identities",
            }[action_value.get()]
            parameters = {}
            if operation_type in {
                    "delete_identity_interval", "expand_identities"}:
                start_frame = int(start_frame_value.get())
                end_frame = int(end_frame_value.get())
                parameters = {
                    "start_imagej_frame": start_frame,
                    "end_imagej_frame": end_frame,
                }
            if operation_type == "expand_identities":
                radius = float(radius_value.get())
                if not radius > 0:
                    raise ValueError("expansion radius must be positive")
                parameters.update({
                    "radius": radius,
                    "radius_unit": radius_unit_value.get(),
                })
            path = save_filtered_operation_session(
                controller.source_manifest, Path(raw_output),
                previewed["signature"], operation_type, parameters,
                operation_count=controller.position)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot apply filtered operation", str(error),
                                 parent=window)
            return
        window.destroy()
        on_saved(path)

    apply_button = ttk.Button(
        controls, text="Apply to previewed identities and save", command=apply,
        state="disabled")
    apply_button.pack(side="right", padx=(0, 6))
    threshold_entry.focus_set()
