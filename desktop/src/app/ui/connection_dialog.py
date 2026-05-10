"""Connection dialog — backend URL + login/logout for prediction upload.

Opened from the gear icon in the navbar; modal Toplevel.
"""

import threading
import tkinter as tk

from src.app.backend_client import BackendError
from src.core.theme import ModernTheme


class ConnectionDialog(tk.Toplevel):
    def __init__(self, parent, client):
        super().__init__(parent)
        self.client = client

        self.title("Backend Connection")
        self.configure(bg=ModernTheme.SURFACE)
        self.resizable(False, False)
        self.transient(parent)

        body = tk.Frame(self, bg=ModernTheme.SURFACE)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(
            body,
            text="Backend Connection",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_XLARGE, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        tk.Label(
            body,
            text="Log in to stream predictions to the backend.",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL),
        ).pack(anchor="w", pady=(0, 15))

        self.url_var = tk.StringVar(value=self.client.base_url)
        self.username_var = tk.StringVar(value=self.client.username or "")
        self.password_var = tk.StringVar()
        self._add_field(body, "Backend URL", self.url_var, show=None)
        self._add_field(body, "Username", self.username_var, show=None)
        self._add_field(body, "Password", self.password_var, show="•")

        btns = tk.Frame(body, bg=ModernTheme.SURFACE)
        btns.pack(fill=tk.X, pady=(10, 0))

        self.login_btn = tk.Button(
            btns,
            text="Log in",
            command=self._on_login,
            **ModernTheme.button_style("primary"),
        )
        self.login_btn.pack(side=tk.LEFT)

        self.register_btn = tk.Button(
            btns,
            text="Register + Log in",
            command=self._on_register,
            **ModernTheme.button_style("secondary"),
        )
        self.register_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.logout_btn = tk.Button(
            btns,
            text="Log out",
            command=self._on_logout,
            **ModernTheme.button_style("secondary"),
        )
        self.logout_btn.pack(side=tk.LEFT, padx=(8, 0))

        tk.Button(
            btns,
            text="Close",
            command=self.destroy,
            **ModernTheme.button_style("secondary"),
        ).pack(side=tk.RIGHT)

        self.status_label = tk.Label(
            body,
            text="",
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_SECONDARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL),
            wraplength=420,
            justify="left",
        )
        self.status_label.pack(anchor="w", pady=(15, 0))

        self._refresh_status()
        self._center_on(parent)

        self.bind("<Escape>", lambda _: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.focus_set()

    # ---- helpers ------------------------------------------------------

    def _add_field(self, parent, label_text, var, show):
        row = tk.Frame(parent, bg=ModernTheme.SURFACE)
        row.pack(fill=tk.X, pady=4)
        tk.Label(
            row,
            text=label_text,
            bg=ModernTheme.SURFACE,
            fg=ModernTheme.TEXT_PRIMARY,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL),
            width=14,
            anchor="w",
        ).pack(side=tk.LEFT)
        entry = tk.Entry(
            row,
            textvariable=var,
            bg=ModernTheme.BACKGROUND_LIGHT,
            fg=ModernTheme.TEXT_PRIMARY,
            insertbackground=ModernTheme.TEXT_PRIMARY,
            relief="flat",
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_NORMAL),
            width=32,
        )
        if show:
            entry.config(show=show)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(8, 0))

    def _center_on(self, parent) -> None:
        self.update_idletasks()
        try:
            px, py = parent.winfo_rootx(), parent.winfo_rooty()
            pw, ph = parent.winfo_width(), parent.winfo_height()
        except tk.TclError:
            return
        w, h = self.winfo_width(), self.winfo_height()
        x = px + max(0, (pw - w) // 2)
        y = py + max(0, (ph - h) // 3)
        self.geometry(f"+{x}+{y}")

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.config(text=text, fg=color)

    def _refresh_status(self) -> None:
        if self.client.is_authenticated:
            self._set_status(
                f"Logged in as {self.client.username}. Predictions will be uploaded.",
                ModernTheme.SUCCESS,
            )
        else:
            self._set_status(
                "Not logged in. Predictions stay local until you authenticate.",
                ModernTheme.TEXT_SECONDARY,
            )

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for b in (self.login_btn, self.register_btn, self.logout_btn):
            b.config(state=state)

    # ---- actions ------------------------------------------------------

    def _on_login(self) -> None:
        self._apply_url()
        username = self.username_var.get().strip()
        password = self.password_var.get()
        if not username or not password:
            self._set_status("Username and password are required.", ModernTheme.WARNING)
            return
        self._set_status("Logging in…", ModernTheme.INFO)
        self._set_busy(True)
        threading.Thread(
            target=self._do_login, args=(username, password, False), daemon=True
        ).start()

    def _on_register(self) -> None:
        self._apply_url()
        username = self.username_var.get().strip()
        password = self.password_var.get()
        if not username or not password:
            self._set_status("Username and password are required.", ModernTheme.WARNING)
            return
        self._set_status("Registering…", ModernTheme.INFO)
        self._set_busy(True)
        threading.Thread(
            target=self._do_login, args=(username, password, True), daemon=True
        ).start()

    def _on_logout(self) -> None:
        self.client.logout()
        self._refresh_status()

    def _apply_url(self) -> None:
        url = self.url_var.get().strip()
        if url and url != self.client.base_url:
            self.client.set_base_url(url)

    def _do_login(self, username: str, password: str, register_first: bool) -> None:
        try:
            if register_first:
                self.client.register(username, password)
            self.client.login(username, password)
            self.after(0, lambda: (self.password_var.set(""), self._refresh_status()))
        except BackendError as e:
            msg = str(e)
            self.after(0, lambda: self._set_status(f"Login failed: {msg}", ModernTheme.ERROR))
        finally:
            self.after(0, lambda: self._set_busy(False))
