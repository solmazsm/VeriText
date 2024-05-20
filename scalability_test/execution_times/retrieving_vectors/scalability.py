#conducted a scalability test by measuring and comparing the execution times for retrieving vectors from PostgreSQL and MySQL databases across various dataset sizes.

axs[0].set_title('Dataset Sizes < 1000')
for size in sizes_under_1k:
    postgres_times = [result[2] for result in postgres_results if result[0] == size]
    mysql_times = [result[2] for result in mysql_results if result[0] == size]
    axs[0].plot(range(1, len(postgres_times) + 1), postgres_times, marker='o', label=f'PostgreSQL {size}')
    axs[0].plot(range(1, len(mysql_times) + 1), mysql_times, marker='o', label=f'MySQL {size}')
axs[0].set_xlabel('Test Run')
axs[0].set_ylabel('Execution Time (seconds)')
axs[0].set_xticks(range(1, len(postgres_times) + 1))
axs[0].grid(True)
axs[0].legend()


axs[1].set_title('Dataset Sizes < 2000')
for size in sizes_under_2k:
    postgres_times = [result[2] for result in postgres_results if result[0] == size]
    mysql_times = [result[2] for result in mysql_results if result[0] == size]
    axs[1].plot(range(1, len(postgres_times) + 1), postgres_times, marker='o', label=f'PostgreSQL {size}')
    axs[1].plot(range(1, len(mysql_times) + 1), mysql_times, marker='o', label=f'MySQL {size}')
axs[1].set_xlabel('Test Run')
axs[1].set_ylabel('Execution Time (seconds)')
axs[1].set_xticks(range(1, len(postgres_times) + 1))
axs[1].grid(True)
axs[1].legend()


axs[2].set_title('Dataset Sizes < 3000')
for size in sizes_under_3k:
    postgres_times = [result[2] for result in postgres_results if result[0] == size]
    mysql_times = [result[2] for result in mysql_results if result[0] == size]
    axs[2].plot(range(1, len(postgres_times) + 1), postgres_times, marker='o', label=f'PostgreSQL {size}')
    axs[2].plot(range(1, len(mysql_times) + 1), mysql_times, marker='o', label=f'MySQL {size}')
axs[2].set_xlabel('Test Run')
axs[2].set_ylabel('Execution Time (seconds)')
axs[2].set_xticks(range(1, len(postgres_times) + 1))
axs[2].grid(True)
axs[2].legend()


axs[3].set_title('Dataset Sizes 8000, 16000, 32000')
for size in sizes_8k_16k_32k:
    postgres_times = [result[2] for result in postgres_results if result[0] == size]
    mysql_times = [result[2] for result in mysql_results if result[0] == size]
    axs[3].plot(range(1, len(postgres_times) + 1), postgres_times, marker='o', label=f'PostgreSQL {size}')
    axs[3].plot(range(1, len(mysql_times) + 1), mysql_times, marker='o', label=f'MySQL {size}')
axs[3].set_xlabel('Test Run')
axs[3].set_ylabel('Execution Time (seconds)')
axs[3].set_xticks(range(1, len(postgres_times) + 1))
axs[3].grid(True)
axs[3].legend()

