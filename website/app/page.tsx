import { Encryption } from "@/components/main/encryption";
import { Hero } from "@/components/main/hero";
import { Skills } from "@/components/main/skills";
import SkyApp from "@/components/sky/SkyApp";

export default function Home() {
  return (
    <main className="h-full w-full">
      <div className="flex flex-col gap-20">
        <Hero />
        <Skills />
        <Encryption />
        <SkyApp />
      </div>
    </main>
  );
}
