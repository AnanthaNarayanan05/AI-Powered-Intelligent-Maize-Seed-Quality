/* ============================================================
   3D maize kernel hero.

   Geometry is procedural (a deformed sphere), so there is no model file to
   download and nothing to decode — the whole scene is a few KB of maths. The
   kernel rotates slowly under warm studio lighting with a scanning ring and a
   sparse point field, to read as "a specimen under analysis" rather than a toy.

   Performance: dpr is capped at 2, the point field is small, there is no
   post-processing, and the frame loop is paused when the tab is hidden. On
   prefers-reduced-motion the canvas is not mounted at all and a static CSS
   fallback renders instead.
   ============================================================ */
import { Suspense, useMemo, useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = (e) => setReduced(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

/** A maize kernel: sphere squashed into the broad teardrop of a dent-corn seed. */
function KernelMesh() {
  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(1, 64, 48);
    const pos = geo.attributes.position;
    const v = new THREE.Vector3();

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);

      // Taper toward the tip cap, flare at the crown.
      const t = (v.y + 1) / 2;
      const taper = 0.55 + 0.65 * Math.pow(t, 0.75);
      v.x *= taper;
      v.z *= taper * 0.72;
      v.y *= 1.22;

      // The dent: a shallow depression in the crown.
      if (v.y > 0.55) {
        const d = (v.y - 0.55) / 0.75;
        v.y -= d * d * 0.34;
      }

      // Fine surface irregularity so it does not read as a plastic primitive.
      const n =
        Math.sin(v.x * 9.3) * Math.cos(v.z * 7.7) * 0.012 +
        Math.sin(v.y * 14.1) * 0.007;
      v.multiplyScalar(1 + n);

      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    return geo;
  }, []);

  const ref = useRef();
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.24; // slow, never spinning
  });

  return (
    <mesh ref={ref} geometry={geometry} castShadow>
      <meshStandardMaterial
        color="#E8B640"
        roughness={0.42}
        metalness={0.12}
        emissive="#3A2607"
        emissiveIntensity={0.28}
      />
    </mesh>
  );
}

/** Analysis ring: a thin torus that sweeps vertically through the specimen. */
function ScanRing() {
  const ref = useRef();
  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.elapsedTime;
    ref.current.position.y = Math.sin(t * 0.55) * 1.35;
    ref.current.material.opacity = 0.28 + Math.sin(t * 0.55 + Math.PI / 2) * 0.16;
  });
  return (
    <mesh ref={ref} rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[1.55, 0.008, 8, 96]} />
      <meshBasicMaterial color="#48D6C6" transparent opacity={0.34} />
    </mesh>
  );
}

/** Sparse orbiting data points — deliberately few, for texture not spectacle. */
function DataPoints({ count = 90 }) {
  const ref = useRef();
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 1.9 + Math.random() * 1.1;
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = (Math.random() - 0.5) * 3.4;
      arr[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    return arr;
  }, [count]);

  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y -= delta * 0.055;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.028} color="#6FBF73" transparent opacity={0.55} sizeAttenuation />
    </points>
  );
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[4, 6, 4]} intensity={2.1} color="#FFE9B8" />
      <directionalLight position={[-5, 1, -3]} intensity={0.6} color="#48D6C6" />
      <pointLight position={[0, -3, 2]} intensity={0.5} color="#6FBF73" />
      <KernelMesh />
      <ScanRing />
      <DataPoints />
    </>
  );
}

export default function SeedHero() {
  const reduced = usePrefersReducedMotion();

  // Static fallback: same visual language, zero animation, no WebGL context.
  if (reduced) {
    return <div className="seed-hero seed-hero--static" aria-hidden="true" />;
  }

  return (
    <div className="seed-hero" aria-hidden="true">
      <Canvas
        camera={{ position: [0, 0, 5.2], fov: 42 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        frameloop="always"
      >
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
      </Canvas>
    </div>
  );
}
