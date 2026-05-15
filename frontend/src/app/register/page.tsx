"use client";

import { Navbar } from "@/components/Navbar";
import { AuthForm } from "@/components/AuthForm";


export default function RegisterPage() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <div className="mx-auto px-6 py-16">
        <AuthForm mode="register" redirectTo="/dashboard" />
      </div>
    </div>
  );
}
