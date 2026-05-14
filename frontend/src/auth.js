const AUTH_TOKEN_KEY = 'medvision.auth.token'
const AUTH_USERNAME_KEY = 'medvision.auth.username'

function hasStorage() {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'
}

export function getAuthToken() {
  if (!hasStorage()) return null
  return localStorage.getItem(AUTH_TOKEN_KEY)
}

export function getStoredAuth() {
  if (!hasStorage()) return { token: null, username: null }
  return {
    token: localStorage.getItem(AUTH_TOKEN_KEY),
    username: localStorage.getItem(AUTH_USERNAME_KEY),
  }
}

export function saveAuth(token, username) {
  if (!hasStorage()) return
  localStorage.setItem(AUTH_TOKEN_KEY, token)
  if (username) {
    localStorage.setItem(AUTH_USERNAME_KEY, username)
  } else {
    localStorage.removeItem(AUTH_USERNAME_KEY)
  }
}

export function clearAuth() {
  if (!hasStorage()) return
  localStorage.removeItem(AUTH_TOKEN_KEY)
  localStorage.removeItem(AUTH_USERNAME_KEY)
}
