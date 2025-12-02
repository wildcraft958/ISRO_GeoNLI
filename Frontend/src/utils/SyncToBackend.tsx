import { useEffect } from "react";
import { useUser } from "@clerk/clerk-react";
import { BACKEND_URL, ROUTES } from "@/lib/constant";

export function SyncUserToBackend() {
  const { user, isLoaded, isSignedIn } = useUser();

  useEffect(() => {
    if (!isLoaded || !isSignedIn || !user) return;

    const sync = async () => {
      await fetch(`${BACKEND_URL}${ROUTES.USER.SYNC}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: user.id,
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
