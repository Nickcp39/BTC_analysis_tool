def main():
    try:
        print("=== Daily BTC update: fetch data + regenerate metrics/plots ===")
    except Exception:
        pass

    from update_btc_price_fred import update_btc_price_fred
    from stepA3_pre_data_summary import main as stepA3_pre_main
    from stepA3_extended_200 import main as stepA3_ext_main
    from stepA3_post_scaled import main as stepA3_post_main
    from stepB4_post_peak_only import main as stepB4_main

    update_btc_price_fred()
    stepA3_pre_main()
    stepA3_ext_main()
    stepA3_post_main()
    stepB4_main()

    try:
        print("All steps finished.")
    except Exception:
        pass


if __name__ == "__main__":
    main()

