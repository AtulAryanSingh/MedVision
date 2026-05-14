const AUTH_TOKEN_KEY = 'medvision.auth.token'
const AUTH_USERNAME_KEY = 'medvision.auth.username'

function hasStorage() {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'
}

function decodeJwtPayload(token) {
  const parts = token.split('.')
  if (parts.length < 2) return null
  try {
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const padLen = (4 - (base64.length % 4)) % 4
    const padded = base64 + '='.repeat(padLen)
    return JSON.parse(atob(padded))
  } catch {
    return null
  }
}

function isTokenUsable(token) {
  if (!token || typeof token !== 'string') return false
  const payload = decodeJwtPayload(token)
  if (!payload) return false
  if (payload.exp === null || payload.exp === undefined) return true
  return payload.exp * 1000 > Date.now()
}

export function getAuthToken() {
  if (!hasStorage()) return null
  return localStorage.getItem(AUTH_TOKEN_KEY)
}

export function getStoredAuth() {
  if (!hasStorage()) return { token: null, username: null }
  const token = localStorage.getItem(AUTH_TOKEN_KEY)
  if (!isTokenUsable(token)) {
    clearAuth()
    return { token: null, username: null }
  }
  return {
    token,
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
