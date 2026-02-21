from engine import run_engine


def main():
    daily_list, hourly_list = run_engine()
    print("DAILY_LIST rows:", len(daily_list))
    print(daily_list.head(50).to_string(index=False))
    print("HOURLY_LIST rows:", len(hourly_list))
    print(hourly_list.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
