/* clients/web/app/api/v1/usage/route.ts
 * Mock GET /api/v1/usage route handler
 * Returns deterministic usage events and an aggregate summary.
 * Supports ?simulate=true to return inflated usage for dev testing.
 */

import { NextResponse } from 'next/server';
import type { UsageEvent, UsageSummary } from '../../../../lib/types';

// GET /api/v1/usage
export async function GET(request: Request) {
  const url = new URL(request.url);
  const simulate = url.searchParams.get('simulate') === 'true';

  // Deterministic base events - stable values for UI/tests
  const baseEvents: UsageEvent[] = [
    {
      id: 'evt_1',
      timestamp: '2025-01-01T00:00:00.000Z',
      projectId: 'proj_1',
      endpoint: '/v1/generate',
      count: 10,
      unitCost: 0.001,
      totalCost: 10 * 0.001,
    },
    {
      id: 'evt_2',
      timestamp: '2025-01-02T12:00:00.000Z',
      projectId: 'proj_1',
      endpoint: '/v1/embeddings',
      count: 5,
      unitCost: 0.002,
      totalCost: 5 * 0.002,
    },
    {
      id: 'evt_3',
      timestamp: '2025-01-03T08:30:00.000Z',
      projectId: 'proj_2',
      endpoint: '/v1/generate',
      count: 2,
      unitCost: 0.001,
      totalCost: 2 * 0.001,
    },
  ];

  const multiplier = simulate ? 3 : 1;

  // Apply multiplier (simulate mode) and normalize numeric precision
  const events: UsageEvent[] = baseEvents.map((e) => {
    const count = e.count * multiplier;
    const totalCost = Number((count * e.unitCost).toFixed(6));
    return { ...e, count, totalCost };
  });

  const summary: UsageSummary = {
    // totalCalls = sum of counts
    totalCalls: events.reduce((acc, ev) => acc + ev.count, 0),
    // cost = sum of totalCost
    cost: Number(events.reduce((acc, ev) => acc + (ev.totalCost ?? 0), 0).toFixed(6)),
  };

  // no-store to keep dev responses fresh
  return NextResponse.json({ events, summary }, { status: 200, headers: { 'Cache-Control': 'no-store' } });
}
