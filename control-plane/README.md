# Axum API Scaffold Template

## 🚀 Quick Start

Install cargo-generate:
```bash
cargo install cargo-generate
```

Generate a new project:
```bash
cargo generate gh:omchillure/axum-api-scaffold
```

It will ask you:
- Project name
- Database name  
- Author name

Then automatically creates the full scaffold! 🎉

## 📦 What You Get

- ✅ Full production architecture (Services, Repositories, Handlers)
- ✅ Axum + Tokio + Diesel setup
- ✅ Database migrations
- ✅ Error handling
- ✅ Validation
- ✅ Password hashing
- ✅ CORS & middleware
- ✅ Structured logging

## 📚 Publishing to GitHub

1. Create a new GitHub repo: `axum-api-scaffold`
2. Push this template:
```bash
git init
git add .
git commit -m "Initial scaffold template"
git remote add origin git@github.com:omchillure/axum-api-scaffold.git
git branch -M main
git push -u origin main
```

3. Now anyone can use:
```bash
cargo generate gh:omchillure/axum-api-scaffold
```