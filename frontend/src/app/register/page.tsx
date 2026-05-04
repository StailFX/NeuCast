"use client";

import { Navbar } from "@/components/Navbar";
import { AuthForm } from "@/components/AuthForm";


export default function RegisterPage() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <Navbar />
      <div className="mx-auto px-6 py-16">
        <AuthForm mode="register" redirectTo="/dashboard" />
      </div>
    </div>
  );
}
