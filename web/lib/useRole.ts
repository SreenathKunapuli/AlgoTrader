"use client";
import { useEffect, useState } from "react";

export type Role = "owner" | "guest" | null;

export function useRole(): Role {
  const [role, setRole] = useState<Role>(null);
  useEffect(() => {
    const match = document.cookie.match(/(?:^|;\s*)lob_role=([^;]+)/);
    setRole((match?.[1] as Role) ?? null);
  }, []);
  return role;
}
