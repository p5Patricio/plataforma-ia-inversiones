-- User-specific risk profiles backed by Supabase Auth.
create table if not exists user_risk_profiles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null default 'default',
  max_position_size numeric not null default 0.10,
  min_confidence_to_trade numeric not null default 0.60,
  max_expected_risk numeric not null default 0.05,
  stop_loss numeric not null default 0.02,
  take_profit numeric not null default 0.04,
  allow_short boolean not null default true,
  is_default boolean not null default true,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(user_id, name)
);

create unique index if not exists user_risk_profiles_one_default_idx
  on user_risk_profiles(user_id)
  where is_default;

alter table user_risk_profiles enable row level security;

drop policy if exists "Users can read own risk profiles" on user_risk_profiles;
create policy "Users can read own risk profiles"
  on user_risk_profiles
  for select
  using (auth.uid() = user_id);

drop policy if exists "Users can insert own risk profiles" on user_risk_profiles;
create policy "Users can insert own risk profiles"
  on user_risk_profiles
  for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can update own risk profiles" on user_risk_profiles;
create policy "Users can update own risk profiles"
  on user_risk_profiles
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
