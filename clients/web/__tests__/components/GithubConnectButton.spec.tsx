import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';

// The component under test is expected at clients/web/components/GithubConnectButton.tsx
// Tests below assert the UI state-machine: connected -> importing -> imported
// and that the component invokes an onImportComplete callback with the imported project payload.
// These tests mock fetch to simulate backend / stubbed responses. If the real component calls
// different endpoints, adapt the mocked responses accordingly.

import GithubConnectButton from '../../../components/GithubConnectButton';

type FetchResponse = any;

function mockFetchSequence(responses: FetchResponse[]) {
  const seq = [...responses];
  const fn = jest.fn().mockImplementation(() => {
    const body = seq.shift();
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(body),
    });
  });
  // @ts-ignore - assign to global for tests
  global.fetch = fn;
  return fn;
}

describe('GithubConnectButton component', () => {
  afterEach(() => {
    // restore fetch to avoid leaking mocks between tests
    // @ts-ignore
    if (global.fetch && (global.fetch as jest.Mock).mockRestore) {
      // jest mocks created by jest.fn() don't have mockRestore; reset instead
      (global.fetch as jest.Mock).mockReset?.();
    }
    // @ts-ignore
    delete global.fetch;
    jest.clearAllMocks();
  });

  test('transitions: connected -> importing -> imported and calls onImportComplete', async () => {
    const onImportComplete = jest.fn();

    // Sequence of mocked backend responses the component is expected to call.
    // Adjust shapes if your component expects different payloads.
    const importResult = { status: 'imported', project: { id: 'proj_1', name: 'Importer Repo' } };
    // First call: starting connection / returning connected state
    // Second call: import finished with project details
    mockFetchSequence([{ status: 'connected' }, importResult]);

    render(<GithubConnectButton onImportComplete={onImportComplete} />);

    // Initial button shows a prompt to connect
    const btn = await screen.findByRole('button');
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveTextContent(/connect github/i);

    // Click to start connect/import flow
    fireEvent.click(btn);

    // After clicking, expect UI to indicate 'connected' or 'importing' (depending on implementation)
    await waitFor(() => expect(screen.getByRole('button')).toHaveTextContent(/importing/i));

    // Wait for the final 'imported' / success state to appear
    await waitFor(() => expect(screen.getByText(/imported/i)).toBeInTheDocument());

    // Ensure callback was invoked with the imported project payload
    expect(onImportComplete).toHaveBeenCalledTimes(1);
    expect(onImportComplete).toHaveBeenCalledWith(expect.objectContaining({ id: 'proj_1', name: 'Importer Repo' }));
  });

  test('uses fallback stub and still finishes import when initial API fails', async () => {
    const onImportComplete = jest.fn();

    // Simulate first fetch failing (network or non-ok) then second returning imported
    const mockFn = jest.fn()
      .mockImplementationOnce(() => Promise.resolve({ ok: false }))
      .mockImplementationOnce(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'imported', project: { id: 'stubbed', name: 'Stub Repo' } }) }));

    // @ts-ignore
    global.fetch = mockFn;

    render(<GithubConnectButton onImportComplete={onImportComplete} />);

    const btn = await screen.findByRole('button');
    fireEvent.click(btn);

    // When the first call fails, component should fall back to a stubbed import path and still progress
    await waitFor(() => expect(screen.getByText(/imported/i)).toBeInTheDocument());

    expect(onImportComplete).toHaveBeenCalledWith(expect.objectContaining({ id: 'stubbed', name: 'Stub Repo' }));
  });

  test('disables button while importing to prevent duplicate actions', async () => {
    const onImportComplete = jest.fn();

    // Make the import call take a little time so we can assert disabled state
    let resolveSecond: () => void;
    const secondPromise = new Promise((res) => (resolveSecond = res));

    const mockFn = jest.fn()
      .mockImplementationOnce(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'connected' }) }))
      .mockImplementationOnce(() => Promise.resolve({ ok: true, json: () => secondPromise }));

    // @ts-ignore
    global.fetch = mockFn;

    render(<GithubConnectButton onImportComplete={onImportComplete} />);

    const btn = await screen.findByRole('button');
    fireEvent.click(btn);

    // After clicking, the button should become disabled while the import is in-flight
    await waitFor(() => expect(btn).toBeDisabled());

    // Resolve the pending import
    // @ts-ignore
    resolveSecond();

    await waitFor(() => expect(onImportComplete).toHaveBeenCalled());

    // After completion, button should be enabled again (or replaced by final state)
    expect(btn.disabled).toBe(false);
  });
});
