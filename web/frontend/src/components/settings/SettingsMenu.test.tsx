// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useDialogs, useLog, useSettings, useSession } from '../../stores'

vi.mock('../../services/socket', () => ({ emitC: vi.fn() }))

import { emitC } from '../../services/socket'
import { SettingsMenu } from './SettingsMenu'

const initialDialogs = useDialogs.getState()
const initialLog = useLog.getState()
const initialSettings = useSettings.getState()
const initialSession = useSession.getState()

beforeEach(() => {
  useDialogs.setState(initialDialogs, true)
  useLog.setState(initialLog, true)
  useSettings.setState(initialSettings, true)
  useSession.setState(initialSession, true)
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('provider and voice settings behavior', () => {
  it('does not contact Codex when another provider is selected', () => {
    useDialogs.getState().setProvider({ provider: 'openai' })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    expect(emitC).not.toHaveBeenCalledWith('get_codex_status', undefined)
  })

  it('times out an unanswered endpoint probe and accepts a successful retry', () => {
    vi.useFakeTimers()
    useDialogs.getState().setProvider({ provider: 'lmstudio' })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.change(screen.getByLabelText('Server URL'), { target: { value: 'http://localhost:1234/v1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Test Connection' }))
    act(() => vi.advanceTimersByTime(30000))
    expect(screen.getByRole('status').textContent).toContain('No test response received')
    expect((screen.getByRole('button', { name: 'Test Connection' }) as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Test Connection' }))
    act(() => useDialogs.getState().setEndpointTestResult({ ok: true, detail: 'Retry accepted' }))
    expect(screen.getByRole('status').textContent).toContain('PASS: Retry accepted')
  })
  it('does not present a closed panel’s late or cached endpoint result on reopen', () => {
    useDialogs.getState().setProvider({ provider: 'lmstudio' })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.change(screen.getByLabelText('Server URL'), { target: { value: 'http://localhost:1234/v1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Test Connection' }))
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    act(() => useDialogs.getState().setEndpointTestResult({ ok: true, detail: 'Old panel result' }))
    expect(screen.queryByRole('status')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    expect(screen.queryByRole('status')).toBeNull()
  })
  it('clears synthetic key inputs after save and removes unsaved keys on close', () => {
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.change(screen.getByLabelText('OpenAI API key'), { target: { value: 'synthetic-only-key' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save Key' }))
    expect((screen.getByLabelText('OpenAI API key') as HTMLInputElement).value).toBe('')
    fireEvent.change(screen.getByLabelText('OpenAI API key'), { target: { value: 'unsaved-synthetic-key' } })
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    expect(document.querySelector('input[type=password]')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    expect((screen.getByLabelText('OpenAI API key') as HTMLInputElement).value).toBe('')
  })
  it('unlocks a provider selection lost during disconnect and permits confirmed retry', () => {
    vi.useFakeTimers()
    useSession.setState({ connected: true })
    useDialogs.getState().setProvider({ provider: 'legacy' })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    const select = screen.getByLabelText('Provider') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'gemini' } })
    act(() => useSession.setState({ connected: false }))
    expect(select.disabled).toBe(true)
    act(() => vi.advanceTimersByTime(10000))
    expect(select.value).toBe('legacy')
    act(() => useSession.setState({ connected: true }))
    fireEvent.change(select, { target: { value: 'openai' } })
    act(() => useDialogs.getState().setProvider({ provider: 'openai' }))
    expect(select.disabled).toBe(false)
    expect(select.value).toBe('openai')
  })
  it('re-enables provider selection when the server never confirms a change', () => {
    vi.useFakeTimers()
    useDialogs.getState().setProvider({ provider: 'legacy' })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    const select = screen.getByLabelText('Provider') as HTMLSelectElement

    fireEvent.change(select, { target: { value: 'openai' } })
    expect(emitC).toHaveBeenCalledWith('set_model_provider', { provider: 'openai' })
    expect(select.disabled).toBe(true)

    act(() => vi.advanceTimersByTime(10000))
    expect(select.disabled).toBe(false)
    expect(select.value).toBe('legacy')
  })

  it('stops and releases an OpenAI preview when settings closes', async () => {
    const pause = vi.fn()
    const play = vi.fn(async () => undefined)
    const revokeObjectURL = vi.fn()
    class CompatibleAudio {
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      pause = pause
      play = play
      src: string
      constructor(src: string) { this.src = src }
    }
    useSettings.setState({ ttsEnabled: true, engine: 'openai' })
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, blob: async () => new Blob(['voice']) })))
    vi.stubGlobal('Audio', CompatibleAudio)
    vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL })

    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.click(screen.getByTitle('Preview voice'))
    await waitFor(() => expect(play).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    expect(pause).toHaveBeenCalledOnce()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:preview')
  })

  it('uses the legacy voice-control geometry hooks and playing state', async () => {
    class CompatibleAudio {
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      pause = vi.fn()
      play = vi.fn(async () => undefined)
      constructor(_src: string) {}
    }
    useSettings.setState({ ttsEnabled: true, engine: 'openai' })
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, blob: async () => new Blob(['voice']) })))
    vi.stubGlobal('Audio', CompatibleAudio)
    vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL: vi.fn() })

    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    const preview = screen.getByTitle('Preview voice')
    expect(preview.parentElement?.classList.contains('neq-voice-controls-parity')).toBe(true)

    fireEvent.click(preview)
    await waitFor(() => expect(screen.getByTitle('Stop preview').classList.contains('playing')).toBe(true))
  })

  it('uses a legacy justify-between row for local Save and Test Connection', () => {
    useDialogs.getState().setProvider({ provider: 'lmstudio' })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))

    const save = screen.getByRole('button', { name: /^Save$/ })
    expect(save.parentElement?.classList.contains('neq-settings-button-row-parity')).toBe(true)
    expect(screen.getByRole('button', { name: 'Test Connection' }).parentElement).toBe(save.parentElement)
  })

  it('uses Ember status tokens with the legacy color fallbacks', () => {
    useDialogs.getState().setProvider({ provider: 'lmstudio' })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.change(screen.getByLabelText('Server URL'), { target: { value: 'http://localhost:1234/v1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Test Connection' }))

    expect(screen.getByRole('status').style.color).toBe('var(--ember-status-pending, #888)')

    act(() => useDialogs.getState().setEndpointTestResult({ ok: true, detail: 'Connected' }))
    expect(screen.getByRole('status').style.color).toBe('var(--ember-status-ok, #2e7d32)')

    fireEvent.click(screen.getByRole('button', { name: 'Test Connection' }))
    act(() => useDialogs.getState().setEndpointTestResult({ ok: false, detail: 'Unavailable' }))
    expect(screen.getByRole('status').style.color).toBe('var(--ember-status-fail, #c62828)')
  })

  it('shows the legacy Auto-play Voice explanation below the hovered row', () => {
    useSettings.setState({ ttsEnabled: true })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    const row = screen.getByText('Auto-play').parentElement as HTMLDivElement
    vi.spyOn(row, 'getBoundingClientRect').mockReturnValue({
      x: 100, y: 200, left: 100, top: 200, right: 380, bottom: 226,
      width: 280, height: 26, toJSON: () => ({}),
    })

    fireEvent.mouseEnter(row)
    const tooltip = screen.getByRole('tooltip')
    expect(screen.getByText('Auto-play Voice')).toBeTruthy()
    expect(tooltip.getAttribute('style')).toContain('left: 100px')
    expect(tooltip.getAttribute('style')).toContain('top: 231px')

    fireEvent.mouseLeave(row)
    expect(screen.queryByRole('tooltip')).toBeNull()
  })

  it('renders discovered Codex models and saves independent tier routing', () => {
    useDialogs.getState().setProvider({ provider: 'codex_oauth' })
    useDialogs.getState().setCodexStatus({
      installed: true,
      authenticated: true,
      email: 'player@example.com',
      plan_type: 'plus',
      models: [
        {
          id: 'account-default',
          display_name: 'Account Default',
          supported_reasoning_efforts: ['low', 'medium', 'high'],
          default_reasoning_effort: 'medium',
          is_default: true,
        },
        {
          id: 'account-strong',
          display_name: 'Account Strong',
          supported_reasoning_efforts: ['high'],
          default_reasoning_effort: 'high',
          is_default: false,
        },
      ],
      routes: { cheap: '', balanced: '', strong: 'account-strong', premium: '' },
      auto_replace_unavailable: true,
      last_fallback: null,
      error: null,
    })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))

    expect(screen.getByText(/Signed in as player@example.com/)).toBeTruthy()
    expect(screen.getByText(/Automatic routing matches each call site's OpenAI model/)).toBeTruthy()
    expect(screen.getAllByRole('option', { name: 'Automatic (match OpenAI routing)' })).toHaveLength(4)
    expect((screen.getByLabelText('Strong tasks') as HTMLSelectElement).value).toBe('account-strong')
    fireEvent.change(screen.getByLabelText('Cheap tasks'), { target: { value: 'account-default' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save Codex Routing' }))
    expect(emitC).toHaveBeenCalledWith('set_codex_routing', {
      routes: {
        cheap: 'account-default', balanced: '', strong: 'account-strong', premium: '',
      },
      auto_replace_unavailable: true,
    })
  })

  it('starts Codex device login and displays the returned code', () => {
    useDialogs.getState().setProvider({ provider: 'codex_oauth' })
    useDialogs.getState().setCodexStatus({
      installed: true,
      authenticated: false,
      models: [],
      routes: { cheap: '', balanced: '', strong: '', premium: '' },
      auto_replace_unavailable: true,
      error: null,
    })
    render(<SettingsMenu />)
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Sign in with ChatGPT' }))
    expect(emitC).toHaveBeenCalledWith('codex_login', undefined)

    act(() => useDialogs.getState().setCodexLogin({
      login_id: 'login-1',
      verification_url: 'https://auth.openai.com/codex/device',
      user_code: 'ABCD-1234',
    }))
    expect(screen.getByText('ABCD-1234')).toBeTruthy()
    expect(screen.getByRole('link').getAttribute('href')).toBe('https://auth.openai.com/codex/device')
  })
})
