/**
 * Visual skeleton placeholder — pulses during loading instead of
 * showing static "—". Sizing is up to the caller via Tailwind classes
 * (h-X w-Y rounded etc.). The animation classes live in globals.css
 * (``.skeleton``).
 */
export function Skeleton({
  className = "",
  rounded = "rounded-md",
}: {
  className?: string;
  rounded?: string;
}) {
  return (
    <span
      aria-hidden
      className={`skeleton inline-block ${rounded} ${className}`}
    />
  );
}
