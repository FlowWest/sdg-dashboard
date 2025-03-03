import json
import psycopg2
from sdgtools.readers

def get_connection():
    return psycopg2.connect(
        host=db_secrets["host"],
        database=db_secrets["database"],
        user=db_secrets["user"],
        password=db_secrets["password"],
        port=db_secrets["port"]
    )

def update_available_years(scenario_name, available_years):
    """Update the available years for a given scenario."""
    conn = get_connection()
    cur = conn.cursor()

    # Convert list to list
    # Update the table
    cur.execute("""
        UPDATE scenarios
        SET available_years = %s
        WHERE name = %s
    """, (available_years, scenario_name))

    conn.commit()
    cur.close()
    conn.close()


MONTH_NAMES = {
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
    }

