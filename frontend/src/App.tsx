import { Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import { SignIn, SignedIn, RedirectToSignIn, SignedOut } from "@clerk/clerk-react";
import { LandingPage } from "./pages/LandingPage";
import "./App.css";
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route
        path="/geo_nli"
        element={
          <>
            <SignedIn>
              <Home />
            </SignedIn>

            <SignedOut>
              <RedirectToSignIn />
            </SignedOut>
          </>
        }
      />
    </Routes>
  );
}
