import os
import re
import time
import queue
import math
import secrets
import hashlib
import sqlite3
import threading
import datetime
import hmac
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Optional libs
try:
    import tensorflow as tf
    Input = tf.keras.layers.Input
    Sequential = tf.keras.models.Sequential
    LSTM, Dense, Dropout = tf.keras.layers.LSTM, tf.keras.layers.Dense, tf.keras.layers.Dropout
    EarlyStopping = tf.keras.callbacks.EarlyStopping
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
plt.style.use('dark_background')
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DB_PATH = os.path.join(BASE_DIR, 'stock_app_users.db')
# ----------------------------
# DB helpers (enable foreign keys)
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            pw_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            added_at TEXT NOT NULL,
            UNIQUE(user_id, symbol),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()
def hash_password(password: str):
    """Return (salt, hash) using pbkdf2_hmac"""
    salt = secrets.token_hex(16)
    hash_bytes = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 200000)
    return salt, hash_bytes.hex()

def verify_password(salt: str, stored_hash_hex: str, password_attempt: str):
    """Verify password using constant-time compare to avoid timing attacks."""
    test_hash = hashlib.pbkdf2_hmac('sha256', password_attempt.encode('utf-8'), salt.encode('utf-8'), 200000)
    return hmac.compare_digest(test_hash.hex(), stored_hash_hex)

def is_strong_password(pw: str):
    """Return (True, '') if strong; otherwise (False, message)."""
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r'[a-z]', pw):
        return False, "Include a lowercase letter."
    if not re.search(r'[A-Z]', pw):
        return False, "Include an uppercase letter."
    if not re.search(r'\d', pw):
        return False, "Include a number."
    if not re.search(r'[^A-Za-z0-9]', pw):
        return False, "Include a special character (e.g. !@#$)."
    return True, ""
# ----------------------------
# Main App
# ----------------------------
class StockAnalyzerApp:
    def __init__(self, root):
        init_db()
        self.root = root
        self.root.title("AI Stock Research Dashboard — LSTM Edition")
        self.root.geometry("1280x760")
        self.root.config(bg="#121212")
        self.wishlist = []
        self.current_user = None  # (id, username)
        self.ui_queue = queue.Queue()
        self.maximized = False
        # login attempt tracking: username -> (attempt_count, first_attempt_timestamp)
        self.login_attempts = {}
        self.create_widgets()
        self.process_ui_queue()
    def create_widgets(self):
        # Top title + auth controls
        top_frame = tk.Frame(self.root, bg="#121212")
        top_frame.pack(fill=tk.X, pady=6)

        tk.Label(top_frame, text="📊 AI Stock Research Dashboard — LSTM Edition", font=("Arial", 20, "bold"),
                 bg="#121212", fg="white").pack(side=tk.LEFT, padx=8)

        auth_frame = tk.Frame(top_frame, bg="#121212")
        auth_frame.pack(side=tk.RIGHT, padx=8)
        self.user_label = tk.Label(auth_frame, text="Not signed in", bg="#121212", fg="white", font=("Arial", 10))
        self.user_label.pack(side=tk.LEFT, padx=(0,8))
        tk.Button(auth_frame, text="🔐 Sign In", command=self.signin_dialog,
                  bg="#6c5ce7", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=3)
        tk.Button(auth_frame, text="📝 Sign Up", command=self.signup_dialog,
                  bg="#00b894", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=3)
        tk.Button(auth_frame, text="🚪 Sign Out", command=self.signout,
                  bg="#d63031", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=3)

        # Entry & buttons
        entry_frame = tk.Frame(self.root, bg="#121212")
        entry_frame.pack(pady=6)

        self.stock_entry = tk.Entry(entry_frame, font=("Arial", 14), width=28,
                                    bg="#1e1e1e", fg="white", insertbackground="white")
        self.stock_entry.pack(side=tk.LEFT, padx=8)

        tk.Button(entry_frame, text="➕ Add to Wishlist", command=self.add_to_wishlist,
                  bg="#0078D7", fg="white", font=("Arial", 11)).pack(side=tk.LEFT, padx=4)
        tk.Button(entry_frame, text="❌ Remove", command=self.remove_stock,
                  bg="#D9534F", fg="white", font=("Arial", 11)).pack(side=tk.LEFT, padx=4)
        tk.Button(entry_frame, text="🔍 Analyze", command=self.threaded_analyze,
                  bg="#5CB85C", fg="white", font=("Arial", 11)).pack(side=tk.LEFT, padx=4)
        tk.Button(entry_frame, text="📄 Export Report", command=self.export_selected_stock,
                  bg="#FFC107", fg="black", font=("Arial", 11)).pack(side=tk.LEFT, padx=4)
        # Left panel: wishlist + live signals
        self.left_frame = tk.Frame(self.root, bg="#121212")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(self.left_frame, text="📋 Wishlist", font=("Arial", 14, "bold"),
                 bg="#121212", fg="white").pack(anchor='nw')
        self.wishlist_box = tk.Listbox(self.left_frame, width=26, height=20, font=("Arial", 12),
                                       bg="#1e1e1e", fg="white", activestyle='none')
        self.wishlist_box.pack(padx=8, pady=6, fill='y')

        tk.Label(self.left_frame, text="🕒 Live Technical Signals (5-min)", font=("Arial", 11, "bold"),
                 bg="#121212", fg="cyan").pack(pady=(10,4))
        self.live_frame = tk.Frame(self.left_frame, bg="#121212", height=140)
        self.live_frame.pack(fill="x", padx=8)
        self.signal_text = tk.Text(self.live_frame, height=7, width=26,
                                   font=("Consolas", 9), bg="#1e1e1e", fg="lightgreen")
        self.signal_text.pack(fill='both', expand=True, padx=2, pady=2)
        # Right panel: info + charts
        right_frame = tk.Frame(self.root, bg="#121212")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.info_text = tk.Text(right_frame, wrap=tk.WORD, width=85, height=12,
                                 font=("Consolas", 11), bg="#1e1e1e", fg="white")
        self.info_text.pack(fill=tk.BOTH, expand=False)
        # Chart area with tabs
        self.chart_frame = tk.Frame(right_frame, bg="#121212")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=(8,0))

        chart_top_bar = tk.Frame(self.chart_frame, bg="#121212")
        chart_top_bar.pack(fill='x')
        # maximize button
        self.max_btn = tk.Button(chart_top_bar, text="⛶ Maximize", command=self.toggle_maximize_chart,
                                 bg="#2d3436", fg="white", relief='raised')
        self.max_btn.pack(side='right', padx=6, pady=2)
        # tab content holder
        self.tab_content_holder = tk.Frame(self.chart_frame, bg="#121212")
        self.tab_content_holder.pack(fill='both', expand=True)
        # create notebook and pages (only once)
        self.tab_control = ttk.Notebook(self.tab_content_holder)
        self.candlestick_tab = ttk.Frame(self.tab_control)
        self.technical_tab = ttk.Frame(self.tab_control)
        self.fundamental_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.candlestick_tab, text='Candlestick')
        self.tab_control.add(self.technical_tab, text='Technical')
        self.tab_control.add(self.fundamental_tab, text='Fundamentals')
        self.tab_control.pack(expand=1, fill='both')
        # start live signals updates
        self.root.after(5000, self.update_live_signals)
    # ----------------------------
    # UI helpers: rendering + scrollable embedding
    # ----------------------------
    def clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()
    def render_fig_to_tab_scrollable(self, fig, tab):
        """Embed a matplotlib Figure into a scrollable area inside given tab frame."""
        self.clear_tab(tab)
        canvas = tk.Canvas(tab, bg="#121212", highlightthickness=0)
        v_scroll = tk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)

        inner = tk.Frame(canvas, bg="#121212")
        inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        canvas.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")

        mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
        mpl_canvas.draw()
        mpl_widget = mpl_canvas.get_tk_widget()
        mpl_widget.pack(fill='both', expand=True)
        def _on_frame_config(event):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass
        inner.bind("<Configure>", _on_frame_config)
        def _on_tab_resize(event):
            try:
                canvas.itemconfig(inner_id, width=event.width - v_scroll.winfo_width())
            except Exception:
                pass
        tab.bind("<Configure>", _on_tab_resize)
        # Mousewheel scroll binding bound to canvas (safer than bind_all)
        def _on_mousewheel(event):
            delta = 0
            if hasattr(event, 'delta'):
                # Windows/macOS
                delta = -1 if event.delta < 0 else 1
            else:
                # Linux Button-4/5
                if event.num == 5:
                    delta = 1
                elif event.num == 4:
                    delta = -1
            try:
                canvas.yview_scroll(delta, "units")
            except Exception:
                pass
        canvas.bind("<Enter>", lambda e: canvas.focus_set())
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)
    def toggle_maximize_chart(self):
        if not hasattr(self, 'left_frame'):
            return
        if not self.maximized:
            try:
                self.left_frame.pack_forget()
            except Exception:
                pass
            self.maximized = True
            self.max_btn.config(text="🗗 Restore")
        else:
            try:
                self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
            except Exception:
                pass
            self.maximized = False
            self.max_btn.config(text="⛶ Maximize")
    # ----------------------------
    # Background UI queue processing (safe)
    # ----------------------------
    def process_ui_queue(self):
        """Process messages from background threads to update the UI safely."""
        try:
            while True:
                task = self.ui_queue.get_nowait()
                if callable(task):
                    try:
                        task()
                    except Exception:
                        pass
                else:
                    widget, text = task
                    try:
                        widget.insert(tk.END, text)
                    except Exception:
                        pass
        except queue.Empty:
            try:
                self.root.after(100, self.process_ui_queue)
            except tk.TclError:
                # app is closing; ignore scheduling
                pass

    # ----------------------------
    # Wishlist / DB operations with validation
    # ----------------------------
    def add_to_wishlist(self):
        stock = self.stock_entry.get().upper().strip()
        if not stock:
            messagebox.showwarning("Input Error", "Please enter a stock symbol.")
            return

        # basic symbol format: letters, numbers, dots, underscores, dashes; 1-10 chars
        if not re.match(r'^[A-Z0-9._-]{1,10}$', stock):
            messagebox.showwarning("Invalid symbol", "Symbol format looks invalid.")
            return

        # quick check with yfinance to ensure ticker exists (small history)
        try:
            test = yf.Ticker(stock).history(period="7d", interval="1d")
            test = test.ffill().dropna()
            if test.empty or len(test) < 1:
                messagebox.showerror("Not found", "Ticker not found or has no recent data.")
                return
        except Exception as e:
            messagebox.showerror("YFinance error", f"Could not verify ticker: {e}")
            return

        if self.current_user is None:
            if stock not in self.wishlist:
                self.wishlist.append(stock)
                self.wishlist_box.insert(tk.END, stock)
                self.stock_entry.delete(0, tk.END)
                messagebox.showinfo("Added (temp)", "Added to wishlist locally. Sign in to save permanently.")
            else:
                messagebox.showinfo("Duplicate", "Stock already in wishlist!")
            return

        user_id = self.current_user[0]
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        c = conn.cursor()
        try:
            c.execute("INSERT OR IGNORE INTO wishlist (user_id, symbol, added_at) VALUES (?, ?, ?)",
                      (user_id, stock, datetime.datetime.now().isoformat()))
            conn.commit()
            if stock not in self.wishlist:
                self.wishlist.append(stock)
                self.wishlist_box.insert(tk.END, stock)
            self.stock_entry.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("DB Error", f"Failed to add to wishlist: {e}")
        finally:
            conn.close()

    def remove_stock(self):
        selected = self.wishlist_box.curselection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a stock to remove.")
            return
        idx = selected[0]
        stock = self.wishlist_box.get(idx)
        if self.current_user:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
            c = conn.cursor()
            try:
                c.execute("DELETE FROM wishlist WHERE user_id=? AND symbol=?", (self.current_user[0], stock))
                conn.commit()
            except Exception as e:
                messagebox.showerror("DB Error", f"Failed to remove: {e}")
            finally:
                conn.close()

        try:
            if stock in self.wishlist:
                self.wishlist.remove(stock)
        except Exception:
            pass
        self.wishlist_box.delete(idx)
        messagebox.showinfo("Removed", f"{stock} removed from wishlist.")

    def load_user_wishlist(self, user_id):
        self.wishlist = []
        self.wishlist_box.delete(0, tk.END)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        c = conn.cursor()
        try:
            c.execute("SELECT symbol FROM wishlist WHERE user_id=? ORDER BY added_at DESC", (user_id,))
            rows = c.fetchall()
            for r in rows:
                sym = r[0]
                self.wishlist.append(sym)
                self.wishlist_box.insert(tk.END, sym)
        finally:
            conn.close()

    # ----------------------------
    # Live signals (intraday)
    # ----------------------------
    def update_live_signals(self):
        """Fetch latest 5-min data for wishlist and show signals."""

        def fetch_and_render():
            try:
                self.signal_text.config(state=tk.NORMAL)
                self.signal_text.delete('1.0', tk.END)
                if not self.wishlist:
                    self.ui_queue.put((self.signal_text, "No stocks in wishlist. Add some to start tracking...\n"))
                else:
                    for symbol in self.wishlist:
                        try:
                            data = yf.download(tickers=symbol, period="1d", interval="5m",
                                               progress=False, auto_adjust=False)
                            data = data.ffill().dropna()
                            if len(data) < 2:
                                self.ui_queue.put((self.signal_text, f"{symbol}: Not enough 5m data\n"))
                                continue

                            data["EMA25"] = data["Close"].ewm(span=20, adjust=False).mean()
                            data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

                            delta = data["Close"].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                            RS = gain / loss if loss.iloc[-1] != 0 else np.inf
                            data["RSI"] = 100 - (100 / (1 + RS))

                            latest = data.iloc[-1]
                            prev = data.iloc[-2]
                            time_str = str(latest.name.time())[:5]
                            if prev["EMA25"] < prev["EMA50"] and latest["EMA25"] > latest["EMA50"]:
                                msg = f"{symbol}: 🔼 BUY (Intraday EMA Crossover) at {time_str}\n"
                            elif prev["EMA25"] > prev["EMA50"] and latest["EMA25"] < latest["EMA50"]:
                                msg = f"{symbol}: 🔽 SELL (Intraday EMA Crossover) at {time_str}\n"
                            elif not np.isinf(latest["RSI"]) and latest["RSI"] > 70:
                                msg = f"{symbol}: ⚠️ SELL (RSI Overbought) at {time_str}\n"
                            elif not np.isinf(latest["RSI"]) and latest["RSI"] < 30:
                                msg = f"{symbol}: ⚡ BUY (RSI Oversold) at {time_str}\n"
                            else:
                                msg = f"{symbol}: HOLD 🤝 at {time_str}\n"
                            self.ui_queue.put((self.signal_text, msg))
                        except Exception as ex:
                            self.ui_queue.put((self.signal_text, f"{symbol}: Error fetching live data ({ex})\n"))
            finally:
                try:
                    self.signal_text.config(state=tk.DISABLED)
                except Exception:
                    pass

        threading.Thread(target=fetch_and_render, daemon=True).start()
        if self.root.winfo_exists():
            try:
                self.root.after(300000, self.update_live_signals)
            except tk.TclError:
                pass

    # ----------------------------
    # Analyze selected stock (threaded)
    # ----------------------------
    def threaded_analyze(self):
        t = threading.Thread(target=self.analyze_stock, daemon=True)
        t.start()

    def analyze_stock(self):
        selected = self.wishlist_box.curselection()
        if not selected:
            messagebox.showwarning("Selection Error", "Select a stock from wishlist first!")
            return
        stock_symbol = self.wishlist_box.get(selected)
        self.display_stock_info(stock_symbol)

    # ----------------------------
    # Display stock info (main)
    # ----------------------------
    def display_stock_info(self, symbol):
        try:
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.ui_queue.put((self.info_text, f"Fetching data for {symbol}...\n"))

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo", interval="1d")
            hist = hist.ffill().dropna()
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            if hist.empty:
                self.info_text.delete(1.0, tk.END)
                self.ui_queue.put((self.info_text, f"⚠️ No data found for {symbol}\n"))
                return

            # Fundamental summary
            self.info_text.delete(1.0, tk.END)
            long_name = info.get('longName', 'N/A')
            self.ui_queue.put((self.info_text, f"=== {symbol} - {long_name} ===\n\n"))
            self.ui_queue.put((self.info_text, f"📅 Sector: {info.get('sector', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"🏢 Industry: {info.get('industry', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"💰 Market Cap: {info.get('marketCap', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"📊 P/E Ratio: {info.get('trailingPE', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"💵 EPS: {info.get('trailingEps', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"📗 Book Value: {info.get('bookValue', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"💸 Dividend Yield: {info.get('dividendYield', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"💼 Debt to Equity: {info.get('debtToEquity', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"📈 52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"📉 52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"📊 Beta: {info.get('beta', 'N/A')}\n"))
            self.ui_queue.put((self.info_text, f"🧠 Profit Margins: {info.get('profitMargins', 'N/A')}\n\n"))

            # Add Global Indices summary
            self.ui_queue.put((self.info_text, "🌍 Global Indices:\n"))
            global_indices = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}
            for s_i, name_i in global_indices.items():
                try:
                    gdata = yf.Ticker(s_i).history(period="5d").ffill().dropna()
                    if not gdata.empty and len(gdata) > 1:
                        last = round(gdata['Close'].iloc[-1], 2)
                        change = round((gdata['Close'].iloc[-1] - gdata['Close'].iloc[-2]) / gdata['Close'].iloc[-2] * 100, 2)
                        self.ui_queue.put((self.info_text, f"{name_i}: {last} ({change}%)\n"))
                except Exception:
                    pass

            # Technical indicators
            hist["EMA25"] = hist["Close"].ewm(span=20, adjust=False).mean()
            hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
            delta = hist["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            RS = gain / loss if loss.iloc[-1] != 0 else np.inf
            hist["RSI"] = 100 - (100 / (1 + RS))
            hist["SMA20"] = hist["Close"].rolling(window=20).mean()
            hist["Upper"] = hist["SMA20"] + (hist["Close"].rolling(window=20).std() * 2)
            hist["Lower"] = hist["SMA20"] - (hist["Close"].rolling(window=20).std() * 2)

            # Show charts (embedded)
            self.ui_queue.put(lambda: self._create_and_show_charts(hist.copy(), info.copy() if info else {}, symbol))

            # LSTM Prediction (if available)
            predicted = None
            if TENSORFLOW_AVAILABLE:
                predicted = self.predict_future_price_lstm(hist.copy(), symbol)
            else:
                self.ui_queue.put((self.info_text, "\n🤖 Prediction: TensorFlow not available — install tensorflow to enable LSTM.\n"))

            # Save analysis snapshot
            def save_snapshot():
                self.last_analysis = {
                    "symbol": symbol,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": self.info_text.get("1.0", tk.END),
                    "hist": hist,
                    "info": info,
                    "prediction": predicted
                }
            self.ui_queue.put(save_snapshot)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch data for {symbol}\n\n{e}")

    def get_technical_recommendation(self, data):
        if data is None or len(data) < 20:
            return "Not enough data for technical analysis"

        data = data.copy().dropna()
        try:
            data["SMA10"] = data["Close"].rolling(window=10, min_periods=1).mean()
            data["SMA20"] = data["Close"].rolling(window=20, min_periods=1).mean()
            data["EMA10"] = data["Close"].ewm(span=10, min_periods=1).mean()
            data["EMA25"] = data["Close"].ewm(span=20, min_periods=1).mean()
            data["RSI"] = self.compute_rsi(data["Close"], 14)
        except Exception:
            return "Error computing indicators"

        latest = data.iloc[-1]
        rsi = latest.get("RSI", 50)
        sma10 = latest.get("SMA10", latest["Close"])
        sma20 = latest.get("SMA20", latest["Close"])
        ema10 = latest.get("EMA10", latest["Close"])
        EMA25 = latest.get("EMA25", latest["Close"])

        signals = []

        if rsi < 35:
            signals.append("BUY (RSI undervalued)")
        elif rsi > 70:
            signals.append("SELL (RSI overbought)")
        else:
            signals.append("HOLD (RSI neutral)")

        if sma10 > sma20 or ema10 > EMA25:
            signals.append("Uptrend detected")
        elif sma10 < sma20:
            signals.append("Downtrend detected")

        if "BUY" in signals[0] and "Uptrend" in " ".join(signals):
            final_signal = "BUY ✅ Strong upward momentum"
        elif "SELL" in signals[0] and "Downtrend" in " ".join(signals):
            final_signal = "SELL 🔻 Weak market signal"
        else:
            final_signal = "HOLD ⚖️ Wait for clearer direction"

        return final_signal

    # ----------------------------
    # Charts (embedded)
    # ----------------------------
    def _create_and_show_charts(self, hist, info, symbol):
        self.ui_queue.put((self.info_text, "\nRendering charts... (embedded tabs)\n"))

        # Candlestick with mplfinance
        try:
            fig_candle, axes = mpf.plot(hist, type='candle', style='charles', mav=(20, 50), volume=True,
                                       title=f"{symbol} - 6 Month Candlestick Chart", warn_too_much_data=10000,
                                       returnfig=True)
            self.render_fig_to_tab_scrollable(fig_candle, self.candlestick_tab)
        except Exception:
            try:
                fig_fallback = Figure(figsize=(8,4), dpi=100)
                ax = fig_fallback.add_subplot(111)
                ax.plot(hist.index, hist['Close'])
                ax.set_title(f"{symbol} - Price (fallback)")
                self.render_fig_to_tab_scrollable(fig_fallback, self.candlestick_tab)
            except Exception:
                pass

        # Technical charts
        try:
            fig_tech = Figure(figsize=(10, 8), dpi=100, constrained_layout=False)

            ax1 = fig_tech.add_subplot(3, 1, 1)
            ax1.plot(hist.index, hist["RSI"])
            ax1.axhline(70, linestyle="--")
            ax1.axhline(30, linestyle="--")
            ax1.set_title(f"{symbol} - RSI")

            ax2 = fig_tech.add_subplot(3, 1, 2, sharex=ax1)
            ax2.plot(hist.index, hist["Close"], label="Close")
            ax2.plot(hist.index, hist["EMA25"], label="EMA25")
            ax2.plot(hist.index, hist["EMA50"], label="EMA50")
            ax2.legend(loc='upper left', fontsize='small')
            ax2.set_title(f"{symbol} - Price & EMA")

            ax3 = fig_tech.add_subplot(3, 1, 3, sharex=ax1)
            ax3.plot(hist.index, hist["Upper"], linestyle="--", label="Upper")
            ax3.plot(hist.index, hist["SMA20"], label="SMA20")
            ax3.plot(hist.index, hist["Lower"], linestyle="--", label="Lower")
            ax3.fill_between(hist.index, hist["Lower"], hist["Upper"], alpha=0.15)
            ax3.legend(loc='upper left', fontsize='small')
            ax3.set_title(f"{symbol} - Bollinger Bands")

            fig_tech.subplots_adjust(hspace=0.35, top=0.95, bottom=0.08)
            for ax in (ax1, ax2, ax3):
                for label in ax.get_xticklabels():
                    label.set_fontsize(8)

            self.render_fig_to_tab_scrollable(fig_tech, self.technical_tab)
        except Exception:
            pass

        # Fundamentals pie
        try:
            labels = ["Market Cap", "Revenue", "Net Income", "Total Assets"]
            values = [
                info.get("marketCap", 0) or 0,
                info.get("totalRevenue", 0) or 0,
                info.get("netIncomeToCommon", 0) or 0,
                info.get("totalAssets", 0) or 0
            ]
            if sum(values) > 0:
                fig_fund = Figure(figsize=(6, 6), dpi=100)
                ax = fig_fund.add_subplot(111)
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title("Fundamental Distribution")
                self.render_fig_to_tab_scrollable(fig_fund, self.fundamental_tab)
            else:
                self.clear_tab(self.fundamental_tab)
        except Exception:
            pass

        # Recommendations appended to info_text
        try:
            intraday = yf.Ticker(symbol).history(period="1d", interval="5m").ffill().dropna()
            short_term = yf.Ticker(symbol).history(period="5d", interval="30m").ffill().dropna()
            weekly = yf.Ticker(symbol).history(period="1mo", interval="1h").ffill().dropna()
            monthly = yf.Ticker(symbol).history(period="6mo", interval="1d").ffill().dropna()
            long_term = hist.copy()

            self.info_text.insert(tk.END, "\n📌 Recommendations:\n")
            self.info_text.insert(tk.END, f"Intraday: {self.get_signal(intraday)}\n")
            self.info_text.insert(tk.END, f"Short-term: {self.get_signal(short_term)}\n")
            self.info_text.insert(tk.END, f"Weekly: {self.get_signal(weekly)}\n")
            self.info_text.insert(tk.END, f"Monthly: {self.get_signal(monthly)}\n")
            self.info_text.insert(tk.END, f"Long-term: {self.get_signal(long_term)}\n")

            indicators = pd.DataFrame()
            indicators['RSI'] = long_term['Close'].rolling(14).apply(
                lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() /
                                            abs(x.diff().clip(upper=0)).mean())) if abs(x.diff().clip(upper=0).mean()) > 1e-9 else 0)
            )
            ema12 = long_term['Close'].ewm(span=12, adjust=False).mean()
            ema26 = long_term['Close'].ewm(span=26, adjust=False).mean()
            indicators['MACD'] = ema12 - ema26
            indicators['Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()

            traders_stop_summary = self.get_traders_stop_metrics(indicators)
            self.info_text.insert(tk.END, f"\n💼 {traders_stop_summary}\n")

        except Exception as e:
            self.info_text.insert(tk.END, f"\nSignal generation error: {e}\n")

    def show_candlestick(self, hist, symbol):
        try:
            fig, _ = mpf.plot(hist, type='candle', style='charles', mav=(20, 50), volume=True,
                              title=f"{symbol} - 6 Month Candlestick Chart", warn_too_much_data=10000,
                              returnfig=True)
            self.render_fig_to_tab_scrollable(fig, self.candlestick_tab)
        except Exception:
            pass

    def show_technical_charts(self, hist, symbol):
        try:
            fig = Figure(figsize=(12, 8))
            ax1 = fig.add_subplot(3,1,1)
            ax1.plot(hist["RSI"])
            ax1.axhline(70, linestyle="--")
            ax1.axhline(30, linestyle="--")
            ax1.set_title(f"{symbol} - RSI")

            ax2 = fig.add_subplot(3,1,2)
            ax2.plot(hist["Close"], label="Close")
            ax2.plot(hist["EMA25"], label="EMA25")
            ax2.plot(hist["EMA50"], label="EMA50")
            ax2.legend()
            ax2.set_title(f"{symbol} - Price & EMA")

            ax3 = fig.add_subplot(3,1,3)
            ax3.plot(hist["Upper"], linestyle="--", label="Upper")
            ax3.plot(hist["SMA20"], label="SMA20")
            ax3.plot(hist["Lower"], linestyle="--", label="Lower")
            ax3.fill_between(hist.index, hist["Lower"], hist["Upper"], alpha=0.15)
            ax3.legend()
            ax3.set_title(f"{symbol} - Bollinger Bands")

            fig.tight_layout()
            self.render_fig_to_tab_scrollable(fig, self.technical_tab)
        except Exception:
            pass

    def show_fundamental_chart(self, info):
        try:
            labels = ["Market Cap", "Revenue", "Net Income", "Total Assets"]
            values = [
                info.get("marketCap", 0) or 0,
                info.get("totalRevenue", 0) or 0,
                info.get("netIncomeToCommon", 0) or 0,
                info.get("totalAssets", 0) or 0
            ]
            if sum(values) == 0:
                return
            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title("Fundamental Distribution")
            self.render_fig_to_tab_scrollable(fig, self.fundamental_tab)
        except Exception:
            pass

    # ----------------------------
    # Prediction (LSTM)
    # ----------------------------
    def create_sequences(self, data, seq_len=60):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def predict_future_price_lstm(self, hist, symbol, seq_len=60, epochs=15, batch_size=8):
        try:
            close = hist["Close"].values.astype('float32')
            if len(close) < seq_len + 10:
                self.ui_queue.put((self.info_text, "\n🤖 LSTM Prediction: not enough historical points (need at least seq_len + 10).\n"))
                return None

            min_val = close.min()
            max_val = close.max()
            scale = max_val - min_val if max_val != min_val else 1.0
            scaled = (close - min_val) / scale

            X, y = self.create_sequences(scaled, seq_len=seq_len)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            tf.keras.backend.clear_session()
            model = Sequential()
            model.add(Input(shape=(seq_len, 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse')

            es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0)
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

            last_seq = scaled[-seq_len:].reshape((1, seq_len, 1))
            pred_scaled = model.predict(last_seq, verbose=0)[0][0]
            pred = pred_scaled * scale + min_val

            self.ui_queue.put((self.info_text, f"\n🤖 LSTM Predicted Next-Day Price: {pred:.2f}\n"))
            return float(pred)
        except Exception as e:
            self.ui_queue.put((self.info_text, f"\nLSTM Prediction Error: {e}\n"))
            return None

    # ==============================
    # SIGNAL RECOMMENDATION ENGINE
    # ==============================
    def get_signal(self, data):
        try:
            if len(data) < 50:
                return "Data Insufficient"

            data["EMA25"] = data["Close"].ewm(span=20, adjust=False).mean()
            data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            data["RSI"] = 100 - (100 / (1 + rs))

            latest = data.iloc[-1]
            EMA25, ema50, rsi = latest["EMA25"], latest["EMA50"], latest["RSI"]

            if EMA25 > ema50 and (not np.isnan(rsi) and rsi < 70):
                return "BUY 🟢"
            elif EMA25 < ema50 and (not np.isnan(rsi) and rsi > 30):
                return "SELL 🔴"
            else:
                return "HOLD ⚪"

        except Exception:
            return "Error"

    def get_traders_stop_metrics(self, indicators):
        try:
            rsi_latest = indicators['RSI'].iloc[-1]
            if rsi_latest < 30:
                rsi_score = 80
            elif rsi_latest > 70:
                rsi_score = 40
            else:
                rsi_score = 100 - abs(rsi_latest - 50)

            macd = indicators['MACD'].iloc[-1]
            signal = indicators['Signal'].iloc[-1]
            macd_hist = macd - signal
            if macd_hist > 0:
                macd_score = 75 + (macd_hist / abs(macd_hist + 1e-5)) * 25
            else:
                macd_score = 50 - (abs(macd_hist) / (abs(macd_hist) + 1)) * 25

            total_score = (rsi_score * 0.5) + (macd_score * 0.5)
            if total_score > 70:
                verdict = "BUY ✅"
            elif total_score < 45:
                verdict = "SELL 🔻"
            else:
                verdict = "HOLD ⚖️"

            return f"TradersStop Score: {total_score:.2f} → {verdict}"

        except Exception as e:
            return f"Error computing proprietary metrics: {e}"

    # ----------------------------
    # Export to PDF / TXT
    # ----------------------------
    def export_selected_stock(self):
        selected = self.wishlist_box.curselection()
        if not selected:
            messagebox.showwarning("Selection Error", "Select a stock first!")
            return
        stock_symbol = self.wishlist_box.get(selected)
        content = getattr(self, "last_analysis", None)
        if content is None:
            messagebox.showwarning("No Data", "Please analyze the stock first before exporting.")
            return

        default_filename = f"{stock_symbol}_Analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        if REPORTLAB_AVAILABLE:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=default_filename + ".pdf",
                                                     filetypes=[("PDF files", "*.pdf")])
            if not file_path:
                return
            try:
                self.export_to_pdf(file_path, content)
                messagebox.showinfo("Exported", f"Report saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export PDF: {e}")
        else:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_filename + ".txt",
                                                     filetypes=[("Text files", "*.txt")])
            if not file_path:
                return
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content.get("text", ""))
                messagebox.showinfo("Exported", f"Text report saved to {file_path}\n(Install reportlab to enable PDF export)")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export text file: {e}")

    def export_to_pdf(self, file_path, content):
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab not available")

        c = canvas.Canvas(file_path, pagesize=letter)
        c.setFont("Helvetica", 10)
        y = 750
        lines = content.get("text", "").splitlines()
        # Header
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Stock Analysis Report — {content.get('symbol', '')}")
        y -= 25
        c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Generated: {content.get('timestamp', '')}")
        y -= 20

        for line in lines:
            if y < 60:
                c.showPage()
                y = 750
                c.setFont("Helvetica", 9)
            # truncate long lines to avoid overflow
            if len(line) > 120:
                line = line[:117] + "..."
            c.drawString(50, y, line)
            y -= 14

        c.save()

    # ----------------------------
    # AUTH DIALOGS (with validation & lockout)
    # ----------------------------
    def signup_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Sign Up")
        dlg.geometry("360x220")
        dlg.config(bg="#1e1e1e")
        dlg.resizable(False, False)

        tk.Label(dlg, text="Create an account", bg="#1e1e1e", fg="white", font=("Arial", 12, "bold")).pack(pady=8)
        tk.Label(dlg, text="Username:", bg="#1e1e1e", fg="white").pack(anchor='w', padx=16)
        username_entry = tk.Entry(dlg)
        username_entry.pack(fill=tk.X, padx=16)

        tk.Label(dlg, text="Password:", bg="#1e1e1e", fg="white").pack(anchor='w', padx=16, pady=(8,0))
        password_entry = tk.Entry(dlg, show='*')
        password_entry.pack(fill=tk.X, padx=16)

        tk.Label(dlg, text="Confirm Password:", bg="#1e1e1e", fg="white").pack(anchor='w', padx=16, pady=(8,0))
        confirm_entry = tk.Entry(dlg, show='*')
        confirm_entry.pack(fill=tk.X, padx=16)

        def do_signup():
            username = username_entry.get().strip()
            pw = password_entry.get()
            pw2 = confirm_entry.get()
            if not username or not pw:
                messagebox.showwarning("Input Error", "Username and password cannot be empty.")
                return
            # username basic validation
            if not re.match(r'^[A-Za-z0-9_.-]{3,30}$', username):
                messagebox.showwarning("Invalid username", "Username must be 3-30 chars: letters, numbers, . _ - allowed.")
                return
            if pw != pw2:
                messagebox.showwarning("Mismatch", "Passwords do not match.")
                return
            ok, msg = is_strong_password(pw)
            if not ok:
                messagebox.showwarning("Weak password", msg)
                return
            salt, pw_hash = hash_password(pw)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
            c = conn.cursor()
            try:
                c.executemany("INSERT INTO users (username, salt, pw_hash, created_at) VALUES (?, ?, ?, ?)",
                              [(username, salt, pw_hash, datetime.datetime.now().isoformat())])
                conn.commit()
                messagebox.showinfo("Success", "Account created. You can sign in now.")
                dlg.destroy()
            except sqlite3.IntegrityError:
                messagebox.showerror("Exists", "Username already exists. Choose another.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create account: {e}")
            finally:
                conn.close()

        tk.Button(dlg, text="Create Account", command=do_signup, bg="#00b894", fg="white").pack(pady=12)

    def signin_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Sign In")
        dlg.geometry("340x180")
        dlg.config(bg="#1e1e1e")
        dlg.resizable(False, False)

        tk.Label(dlg, text="Welcome back", bg="#1e1e1e", fg="white", font=("Arial", 12, "bold")).pack(pady=8)
        tk.Label(dlg, text="Username:", bg="#1e1e1e", fg="white").pack(anchor='w', padx=16)
        username_entry = tk.Entry(dlg)
        username_entry.pack(fill=tk.X, padx=16)

        tk.Label(dlg, text="Password:", bg="#1e1e1e", fg="white").pack(anchor='w', padx=16, pady=(8,0))
        password_entry = tk.Entry(dlg, show='*')
        password_entry.pack(fill=tk.X, padx=16)

        def do_signin():
            username = username_entry.get().strip()
            pw = password_entry.get()
            if not username or not pw:
                messagebox.showwarning("Input Error", "Please enter username and password.")
                return

            # --- simple lockout policy ---
            attempts, first_ts = self.login_attempts.get(username, (0, 0))
            now = time.time()
            # reset attempts older than 300s (5 minutes)
            if attempts and now - first_ts > 300:
                attempts, first_ts = 0, now

            if attempts >= 5:
                messagebox.showerror("Locked Out", "Too many failed attempts. Try again later.")
                return

            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
            c = conn.cursor()
            try:
                c.execute("SELECT id, salt, pw_hash FROM users WHERE username=?", (username,))
                row = c.fetchone()
                if not row:
                    attempts += 1
                    self.login_attempts[username] = (attempts, first_ts or now)
                    messagebox.showerror("Not found", "No such user. Please sign up.")
                    return
                uid, salt, stored_hash = row
                if verify_password(salt, stored_hash, pw):
                    # successful login: reset attempts
                    if username in self.login_attempts:
                        del self.login_attempts[username]
                    self.current_user = (uid, username)
                    self.user_label.config(text=f"Signed in: {username}")
                    messagebox.showinfo("Welcome", f"Signed in as {username}")
                    dlg.destroy()
                    self.load_user_wishlist(uid)
                else:
                    attempts += 1
                    self.login_attempts[username] = (attempts, first_ts or now)
                    remaining = max(0, 5 - attempts)
                    messagebox.showerror("Invalid", f"Wrong password. Attempts left: {remaining}")
            except Exception as e:
                messagebox.showerror("Error", f"Sign-in failed: {e}")
            finally:
                conn.close()

        tk.Button(dlg, text="Sign In", command=do_signin, bg="#6c5ce7", fg="white").pack(pady=12)

    def signout(self):
        if self.current_user is None:
            messagebox.showinfo("Not signed in", "You are not signed in.")
            return
        username = self.current_user[1]
        self.current_user = None
        self.user_label.config(text="Not signed in")
        # clear wishlist from UI
        self.wishlist = []
        self.wishlist_box.delete(0, tk.END)
        messagebox.showinfo("Signed out", f"Signed out {username}")

    # ----------------------------
    # small helper used earlier
    def compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalyzerApp(root)
    root.mainloop()
