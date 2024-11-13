import mysql.connector
from mysql.connector import Error

try:
    # Menghubungkan ke database MySQL
    conn = mysql.connector.connect(
        host='localhost',      # Ganti dengan host jika berbeda
        database='library',    # Pastikan database ini ada atau ganti dengan nama yang diinginkan
        user='root',  # Ganti dengan username MySQL Anda
        password=''  # Ganti dengan password MySQL Anda
    )

    if conn.is_connected():
        print('Connected to MySQL database')

        c = conn.cursor()

        # Membuat tabel users
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL
            )
        ''')

        # Membuat tabel books
        c.execute('''
            CREATE TABLE IF NOT EXISTS books (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                author VARCHAR(255) NOT NULL,
                publication_date DATE NOT NULL,
                language VARCHAR(50) NOT NULL,
                description TEXT
            )
        ''')

        # Membuat tabel loans (peminjaman)
        c.execute('''
            CREATE TABLE IF NOT EXISTS loans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                book_id INT NOT NULL,
                borrow_date DATE NOT NULL,
                return_date DATE,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (book_id) REFERENCES books(id)
            )
        ''')

        # Menambahkan sample data ke tabel users
        users = [
            ('Afina Anfa Ana', 'afina@example.com'),
            ('John Doe', 'john.doe@example.com'),
            ('Alice Johnson', 'alice.j@example.com'),
            ('Bob Brown', 'bob.brown@example.com'),
            ('Catherine Green', 'cathy.green@example.com'),
            ('David White', 'david.white@example.com'),
            ('Eva Black', 'eva.black@example.com'),
            ('Frank Grey', 'frank.grey@example.com'),
            ('Grace Blue', 'grace.blue@example.com'),
            ('Henry Gold', 'henry.gold@example.com'),
        ]

        c.executemany("INSERT INTO users (name, email) VALUES (%s, %s)", users)

        # Menambahkan sample data ke tabel books
        books = [
            ('Metaverse & Digital Twins', 'John Doe', '2023-05-10', 'English', 'Exploring digital twins and metaverse'),
            ('AI Revolution', 'Jane Smith', '2022-08-14', 'English', 'A deep dive into AI technologies'),
            ('Understanding Quantum Computing', 'Alice Johnson', '2021-11-01', 'English', 'Basics of quantum computing'),
            ('The Future of Work', 'Bob Brown', '2020-06-21', 'English', 'How technology is changing jobs'),
            ('Data Science for Beginners', 'Catherine Green', '2019-03-15', 'English', 'An introduction to data science'),
            ('Machine Learning in Action', 'David White', '2022-01-10', 'English', 'Practical applications of ML'),
            ('Blockchain Basics', 'Eva Black', '2023-02-02', 'English', 'Understanding blockchain technology'),
            ('Cybersecurity Essentials', 'Frank Grey', '2021-09-30', 'English', 'Fundamentals of cybersecurity'),
            ('Digital Marketing Strategies', 'Grace Blue', '2022-07-14', 'English', 'Effective digital marketing techniques'),
            ('The Art of Programming', 'Henry Gold', '2023-08-28', 'English', 'Exploring programming languages and paradigms'),
        ]

        c.executemany("INSERT INTO books (title, author, publication_date, language, description) VALUES (%s, %s, %s, %s, %s)", books)

        # Menambahkan sample data ke tabel loans (peminjaman)
        loans = [
            (1, 1, '2024-10-01', '2024-10-15'),
            (2, 2, '2024-09-15', '2024-09-30'),
            (3, 3, '2024-09-20', None),
            (4, 4, '2024-10-05', None),
            (5, 5, '2024-09-10', '2024-09-17'),
            (6, 6, '2024-09-25', None),
            (7, 7, '2024-10-02', '2024-10-09'),
            (8, 8, '2024-10-07', None),
            (9, 9, '2024-10-10', None),
            (10, 10, '2024-10-11', None),
        ]

        c.executemany("INSERT INTO loans (user_id, book_id, borrow_date, return_date) VALUES (%s, %s, %s, %s)", loans)

        # Commit dan close
        conn.commit()

except Error as e:
    print(f"Error: {e}")
finally:
    if conn.is_connected():
        c.close()
        conn.close()
        print('MySQL connection is closed')
