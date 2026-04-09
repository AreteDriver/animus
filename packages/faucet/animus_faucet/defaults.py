"""Default faucet configurations — starter set of known-good faucets."""

DEFAULT_FAUCETS_YAML = """\
# Animus Faucet Configuration
# Edit wallet addresses and enable/disable faucets as needed.

jitter_seconds: 300  # Random delay ±5min on each claim
headless: true
auto_disable_threshold: 10  # Disable faucet after N consecutive failures

proxy:
  enabled: false
  provider: brightdata
  endpoint: ""
  username: ""
  password: ""
  rotation_interval_seconds: 300

vision:
  enabled: false
  ollama_url: http://localhost:11434
  model: llava
  audio_fallback: true

consolidation:
  enabled: false
  schedule_cron: "0 3 * * 0"  # Weekly Sunday 3am
  min_balance_usd: 1.0
  gas_price_max_gwei: 30
  cold_wallet_addresses: {}

alerts:
  discord_webhook_url: ""
  alert_on_ban: true
  alert_on_consolidation: true
  daily_summary: true
  daily_summary_cron: "0 9 * * *"

faucets:
  # --- API Faucets (no captcha, highest ROI) ---

  - name: sui_cli_faucet
    url: "https://faucet.testnet.sui.io/v1/gas"
    chain: sui_testnet
    driver_type: api
    captcha_type: none
    schedule_cron: "0 */2 * * *"  # every 2 hours
    expected_payout_usd: 0.00
    enabled: true
    requires_proxy: false
    notes: "Sui testnet — free SUI for testing. Reliable."
    driver_config:
      method: POST
      payload_template:
        FixedAmountRequest:
          recipient: "{address}"
      success_field: null
      amount_field: null

  - name: sui_cli_local
    url: ""
    chain: sui_testnet
    driver_type: cli
    captcha_type: none
    schedule_cron: "30 */2 * * *"  # offset from API faucet
    expected_payout_usd: 0.00
    enabled: false
    requires_proxy: false
    notes: "Local sui CLI faucet — requires sui client installed"
    driver_config:
      command: "sui client faucet --address {address}"
      success_pattern: "Request successful"
      timeout: 30

  # --- Browser Faucets (captcha required) ---

  - name: freebitcoin
    url: "https://freebitco.in"
    chain: btc
    driver_type: browser
    captcha_type: hcaptcha
    schedule_cron: "0 * * * *"  # hourly
    expected_payout_usd: 0.01
    enabled: false
    requires_proxy: true
    notes: "Oldest BTC faucet (2013). Hourly freeroll. Requires hCaptcha."
    driver_config:
      address_selector: "#btc_address"
      submit_selector: "#free_play_form_button"
      success_selector: ".free_play_result"
      amount_selector: "#free_play_digits"

  - name: firefaucet
    url: "https://firefaucet.win"
    chain: btc
    driver_type: browser
    captcha_type: hcaptcha
    schedule_cron: "15 * * * *"
    expected_payout_usd: 0.005
    enabled: false
    requires_proxy: true
    notes: "Multi-coin faucet. Auto-claim feature."
    driver_config:
      address_selector: "input[name='address']"
      submit_selector: "#claim-button"
      success_selector: ".claim-success"

  - name: cointiply
    url: "https://cointiply.com"
    chain: btc
    driver_type: browser
    captcha_type: recaptcha_v2
    schedule_cron: "30 * * * *"
    expected_payout_usd: 0.01
    enabled: false
    requires_proxy: true
    notes: "Higher payouts via surveys + offers. reCAPTCHA."
    driver_config:
      address_selector: "#withdrawal-address"
      submit_selector: "#claim-btn"
      success_selector: ".success-message"

  # --- Testnet Faucets (for project deployments) ---

  - name: alchemy_sepolia
    url: "https://api.alchemy.com/v1/faucet/sepolia"
    chain: eth
    driver_type: api
    captcha_type: none
    schedule_cron: "0 6 * * *"  # daily
    expected_payout_usd: 0.00
    enabled: false
    requires_proxy: false
    notes: "Sepolia ETH for Base bridging. Requires Alchemy API key in headers."
    driver_config:
      method: POST
      headers:
        Authorization: "Bearer YOUR_ALCHEMY_API_KEY"
      payload_template:
        address: "{address}"
      success_field: "success"

  - name: google_sepolia
    url: "https://cloud.google.com/application/web3/faucet/ethereum/sepolia"
    chain: eth
    driver_type: browser
    captcha_type: none
    schedule_cron: "0 12 * * *"  # daily noon
    expected_payout_usd: 0.00
    enabled: false
    requires_proxy: false
    notes: "Google Cloud faucet — generous, no captcha"
    driver_config:
      address_selector: "input[placeholder*='address']"
      submit_selector: "button[type='submit']"
      success_selector: ".success"
"""
