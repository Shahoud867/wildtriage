import { type FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api/client";

export function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api.login(email, password);
      navigate("/projects");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed");
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-lg border border-stone-200 p-6 dark:border-stone-800">
        <h1 className="text-xl font-semibold">Sign in to WildTriage</h1>
        <label className="block text-sm">
          Email
          <input className="mt-1 w-full rounded border px-2 py-1.5 dark:bg-stone-900" type="email" value={email}
            onChange={(e) => setEmail(e.target.value)} required autoFocus />
        </label>
        <label className="block text-sm">
          Password
          <input className="mt-1 w-full rounded border px-2 py-1.5 dark:bg-stone-900" type="password" value={password}
            onChange={(e) => setPassword(e.target.value)} required />
        </label>
        {error && <p role="alert" className="text-sm text-conf-low">{error}</p>}
        <button className="w-full rounded bg-brand px-3 py-2 text-white" type="submit">Sign in</button>
      </form>
    </div>
  );
}
