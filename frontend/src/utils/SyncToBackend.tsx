import { useEffect } from "react";
import { useUser } from "@clerk/clerk-react";
import { BACKEND_URL, routes } from "@/lib/api";
export function SyncUserToBackend() {
  const { user, isLoaded, isSignedIn } = useUser();

  useEffect(() => {
    if (!isLoaded || !isSignedIn || !user) return;

    const sync = async () => {
      await fetch(`${BACKEND_URL}${routes.USER_SYNC}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clerk_id: user.id,
          email: user.primaryEmailAddress?.emailAddress,
          username: user.firstName,
        }),
        credentials: "include",
      });
    };

    sync();
  }, []);

  // no UI
  return null;
}
