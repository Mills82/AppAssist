/* clients/web/components/ModeChip.tsx
 * Small ModeChip React component used in the chat header.
 * Shows the current mode (e.g. Auto) and toggles an inline hint/advanced summary when clicked.
 */

import React, { FC, useState } from 'react';

export interface ModeChipProps {
  /** Label to show in the chip (e.g. "Auto", "Manual", "Q&A") */
  mode?: string;
  /** Hint text revealed when the chip is toggled open. If omitted a sensible default is used. */
  hint?: string;
  /** Additional tailwind/css classes to apply to the outer container */
  className?: string;
  /** If true the hint is visible initially */
  initiallyOpen?: boolean;
  /** Optional callback when the open state changes */
  onToggle?: (open: boolean) => void;
}

/**
 * ModeChip
 *
 * Simple, accessible chip that displays a small label and reveals a hint/advanced summary
 * when clicked. Uses Tailwind utility classes for styling but keeps markup minimal so it
 * works fine in non-Tailwind setups as well.
 */
const ModeChip: FC<ModeChipProps> = ({
  mode = 'Auto',
  hint,
  className = '',
  initiallyOpen = false,
  onToggle,
}) => {
  const [open, setOpen] = useState<boolean>(initiallyOpen);

  const computedHint = hint ?? (mode === 'Auto'
    ? 'Auto chose: Q&A / Analyze / Plan / Edit'
    : `Mode: ${mode}`);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (onToggle) onToggle(next);
  };

  return (
    <div className={`inline-flex items-center space-x-3 ${className}`}>
      <button
        type="button"
        aria-expanded={open}
        aria-pressed={open}
        onClick={toggle}
        className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-100 rounded-full text-sm hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
      >
        <span className="font-semibold select-none">{mode}</span>
        {/* simple chevron icon */}
        <svg
          className={`w-4 h-4 transform transition-transform duration-150 ${open ? 'rotate-180' : 'rotate-0'}`}
          viewBox="0 0 20 20"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {/* Animated hint area. Keep height collapse via max-width and opacity for a subtle inline reveal. */}
      <div
        className={`flex items-center transition-all duration-150 overflow-hidden ${open ? 'max-w-xs opacity-100' : 'max-w-0 opacity-0'}`}
        aria-hidden={!open}
      >
        <span className="text-xs text-gray-600 dark:text-gray-300 truncate">{computedHint}</span>
      </div>
    </div>
  );
};

export default ModeChip;

// Small demo export that can be rendered inside a story or a test page.
export const ModeChipDemo: FC = () => {
  const [mode, setMode] = useState('Auto');
  const [openCount, setOpenCount] = useState(0);

  return (
    <div className="p-4 space-y-4 bg-white dark:bg-gray-900">
      <div className="flex items-center gap-4">
        <ModeChip
          mode={mode}
          onToggle={(open) => {
            if (open) setOpenCount((c) => c + 1);
          }}
        />

        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
          <label className="sr-only">Mode selector</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="text-sm bg-transparent border border-gray-200 dark:border-gray-700 rounded px-2 py-1"
          >
            <option>Auto</option>
            <option>Q&amp;A</option>
            <option>Analyze</option>
            <option>Plan</option>
            <option>Edit</option>
            <option>Manual</option>
          </select>
          <div className="text-xs text-gray-500">Opened: {openCount}</div>
        </div>
      </div>

      <div className="text-xs text-gray-500">Tip: click the chip to reveal a short inline hint or advanced controls.</div>
    </div>
  );
};

/*
Usage notes:
- Import and render <ModeChip mode="Auto" /> inside your chat header.
- Use the ModeChipDemo in a local dev page or Storybook story to preview behavior.
*/
