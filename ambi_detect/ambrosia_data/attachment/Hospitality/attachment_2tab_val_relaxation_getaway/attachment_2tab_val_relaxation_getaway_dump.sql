BEGIN TRANSACTION;
CREATE TABLE Day_Spa(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Services TEXT,
    Price INTEGER
);
INSERT INTO "Day_Spa" VALUES(1,'Oasis Day Spa','New York','Massage, Facial, Manicure',150);
INSERT INTO "Day_Spa" VALUES(2,'Serenity Day Spa','Los Angeles','Aromatherapy, Yoga',300);
INSERT INTO "Day_Spa" VALUES(3,'Escape Day Spa','Chicago','Hot Stone Massage, Acupuncture',250);
INSERT INTO "Day_Spa" VALUES(4,'Zen Day Spa','Miami','Reflexology, Herbal Wraps',200);
INSERT INTO "Day_Spa" VALUES(5,'Paradise Day Spa','Dallas','Infrared Sauna, Meditation',180);
CREATE TABLE Guest(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber TEXT
);
INSERT INTO "Guest" VALUES(1,'Anna','White','anna@example.com','123-456-7890');
INSERT INTO "Guest" VALUES(2,'Ben','Black','ben@example.com','234-567-8901');
INSERT INTO "Guest" VALUES(3,'Cathy','Green','cathy@example.com','345-678-9012');
INSERT INTO "Guest" VALUES(4,'Daniel','Blue','daniel@example.com','456-789-0123');
INSERT INTO "Guest" VALUES(5,'Elizabeth','Red','elizabeth@example.com','567-890-1234');
CREATE TABLE Hotel(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Rooms TEXT,
    Amenities TEXT
);
INSERT INTO "Hotel" VALUES(1,'Grand Plaza Hotel','New York','300 rooms','Gym, Pool, Restaurant');
INSERT INTO "Hotel" VALUES(2,'Riverview Inn','California','200 rooms','Spa, Bar, Parking');
INSERT INTO "Hotel" VALUES(3,'City Center Suites','Illinois','250 rooms','Business center, Concierge service, Laundry');
INSERT INTO "Hotel" VALUES(4,'Coastal Breeze Resort','Texas','275 rooms','Kids club, Tennis courts, Beach access');
INSERT INTO "Hotel" VALUES(5,'Urban Loft Hotel','Georgia','225 rooms','Free WiFi, Lounge area, Room service');
CREATE TABLE Restaurant(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Cuisine TEXT,
    Menu TEXT
);
INSERT INTO "Restaurant" VALUES(1,'The Gourmet Table','New York','French','Steak frites, Coq au vin, Crème brûlée');
INSERT INTO "Restaurant" VALUES(2,'Asian Delights','California','Japanese','Sushi rolls, Tempura, Soba noodles');
INSERT INTO "Restaurant" VALUES(3,'Italiano Ristorante','Illinois','Italian','Pasta, Risotto, Tiramisu');
INSERT INTO "Restaurant" VALUES(4,'Baja Grill','Texas','Mexican','Enchiladas, Fajitas, Churros');
INSERT INTO "Restaurant" VALUES(5,'Curry House','Georgia','Indian','Butter chicken, Biryani, Samosas');
CREATE TABLE Spa_Therapist(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Specialty TEXT,
    Experience INT
);
INSERT INTO "Spa_Therapist" VALUES(1,'Alice','Smith','Massage',10);
INSERT INTO "Spa_Therapist" VALUES(2,'Bob','Johnson','Yoga',8);
INSERT INTO "Spa_Therapist" VALUES(3,'Carol','Williams','Acupuncture',6);
INSERT INTO "Spa_Therapist" VALUES(4,'David','Brown','Herbal Wraps',12);
INSERT INTO "Spa_Therapist" VALUES(5,'Emily','Jones','Facials',7);
CREATE TABLE Wellness_Retreat(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Activities TEXT,
    Price INTEGER
);
INSERT INTO "Wellness_Retreat" VALUES(1,'Mountain Retreat','Colorado','Hiking, Birdwatching',150);
INSERT INTO "Wellness_Retreat" VALUES(2,'Beachside Retreat','Florida','Surfing, Paddleboarding',300);
INSERT INTO "Wellness_Retreat" VALUES(3,'Forest Retreat','Washington','Camping, Fishing',250);
INSERT INTO "Wellness_Retreat" VALUES(4,'Desert Oasis','Nevada','Stargazing, Offroading',200);
INSERT INTO "Wellness_Retreat" VALUES(5,'Lakeside Escape','Minnesota','Boating, Swimming',180);
COMMIT;
